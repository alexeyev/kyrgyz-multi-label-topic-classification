""" Multilabel classification with non-DL char-ngram-based methods """
import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import jaccard_score, accuracy_score, ndcg_score, zero_one_loss
from sklearn.metrics import make_scorer, hamming_loss, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.adapt import MLkNN, BRkNNaClassifier
from skmultilearn.model_selection import IterativeStratification

if __name__ == "__main__":

    CHAR_LEVEL = True
    THREADS = 4

    logger = logging.getLogger("evaluation-bag-of-ngrams")
    logger.setLevel(logging.DEBUG)
    TOK_OR_CHAR = 'tokens' if not CHAR_LEVEL else 'characters'
    fh = logging.FileHandler(f"classic-eval-bow-{TOK_OR_CHAR}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s\t[%(levelname)s]\t%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    jaccard_avg_samples = make_scorer(jaccard_score, average="samples", greater_is_better=True)


    def at_least_one_hit_rate(y_true: np.ndarray, y_pred):
        """ If at least one predicted label matches, then success """
        if y_pred is not np.ndarray:
            y_pred = y_pred.toarray()
        hits_per_line = np.sum(y_true * y_pred, axis=1)
        return np.sum(hits_per_line > 0) / y_true.shape[0]


    logging.info("Reading all embeddings")

    df_train = pd.read_csv("data/qad-train_1000.csv")
    df_test = pd.read_csv("data/qad-test_0500.csv")

    orig_train_text = df_train["tokenized_text"].map(str)
    orig_test_text = df_test["tokenized_text"].map(str)

    mlb = MultiLabelBinarizer(sparse_output=False)
    orig_y_train = mlb.fit_transform(df_train["labels_set"].map(lambda x: tuple(x.split(","))))
    orig_y_test = mlb.transform(df_test["labels_set"].map(lambda x: tuple(x.split(","))))

    np.random.seed(100)

    stratifier = IterativeStratification(n_splits=2, order=1)

    train_indexes, dev_indexes = next(stratifier.split(list(orig_train_text), orig_y_train))

    logger.info("2-fold split")
    logger.info(f"{train_indexes.shape}, {dev_indexes.shape}")

    cv = [(train_indexes, dev_indexes), (dev_indexes, train_indexes)]

    orig_train_text = np.array(list(orig_train_text))

    n = len(train_indexes)
    base_lr = LogisticRegression(random_state=0)
    base_sgd_log = SGDClassifier(n_jobs=1, random_state=0,
                                 tol=1e-10,
                                 max_iter=np.ceil(10 ** 6 / n))

    classifiers = [
        (MultiOutputClassifier(base_lr), {"estimator__C": [0.7, 0.9, 1.0],
                                          "estimator__penalty": ["l2"],
                                          "estimator__max_iter": [1000, 10000]}),
        (MLkNN(), {"k": [1, 2, 3, 5, 10], "s": [0.1, 0.5, 0.7, 1.0]}),
        (BRkNNaClassifier(), {"k": [1, 2, 3, 5, 10]}),
        (ClassifierChain(base_lr, random_state=0), {"order": [None, "random"],
                                                    "base_estimator__C": [0.7, 0.9, 1.0],
                                                    "base_estimator__penalty": ["l2"]}),
        (ClassifierChain(base_sgd_log, random_state=0), {"order": [None, "random"],
                                                         "base_estimator__average": [True, False],
                                                         "base_estimator__loss": ["hinge",
                                                                                  "log_loss",
                                                                                  "huber"],
                                                         "base_estimator__penalty": ["l1", "l2"],
                                                         "base_estimator__max_iter": [20000, 100000]}),
        (MultiOutputClassifier(base_sgd_log), {"estimator__average": [True, False],
                                               "estimator__loss": ["hinge", "log_loss", "huber"],
                                               "estimator__penalty": ["l1", "l2"],
                                               "estimator__max_iter": [20000, 100000]}),
    ]

    for classifier, classifier_params in classifiers:

        logger.info("--\n")
        logger.info(re.sub(" +", " ", str(classifier).replace("\n", " ")) + "\n")
        vectorizer = CountVectorizer(lowercase=True)
        pipeline = Pipeline(steps=[("vec", vectorizer), ("clf", classifier)])

        if not CHAR_LEVEL:
            param_grid_vec = {"vec__ngram_range": [(1, 1), (1, 2)],
                              "vec__analyzer": ["word"],
                              "vec__max_df": [0.4, 0.6, 0.8, 1.0],
                              "vec__min_df": [2, 5, 10],
                              "vec__max_features": [2000, 5000]
                              }
        else:
            param_grid_vec = {
                "vec__ngram_range": [(2, 3), (3, 4)],
                "vec__analyzer": ["char_wb"],
                "vec__max_df": [0.4, 0.7, 1.0],
                "vec__min_df": [4, 10, 15],
                "vec__max_features": [2000, 5000]
            }

        param_grid_clf = {"clf__" + key: val for key, val in classifier_params.items()}
        param_grid = {**param_grid_vec, **param_grid_clf}

        search = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=THREADS,
                              cv=cv, scoring=jaccard_avg_samples, verbose=1)

        search.fit(orig_train_text, orig_y_train)

        logger.info("Best parameter (CV score=%0.3f):" % search.best_score_)
        logger.info(f"{search.best_params_}")

        best_pipeline = search.best_estimator_
        retrained_pipeline = best_pipeline.fit(orig_train_text, orig_y_train)
        predictions = retrained_pipeline.predict(orig_test_text)
        predictions_proba = None

        try:
            HAVE_PROB_ESTIMATES = True
            predictions_proba = retrained_pipeline.predict_proba(orig_test_text)
            if predictions_proba is not np.ndarray:
                predictions_proba = predictions_proba.toarray()
        except Exception as e:
            HAVE_PROB_ESTIMATES = False
            print(e)

        if HAVE_PROB_ESTIMATES:
            logger.info(
                f"NDCG-3 score      {ndcg_score(orig_y_test, predictions_proba, k=3) * 100:02.1f}% (more is better)")
        else:
            logger.info("NDCG              not computable")

        logger.info(
            f"at-least-1 score  {at_least_one_hit_rate(orig_y_test, predictions) * 100:02.1f}% (more is better)")
        logger.info(
            f"Jaccard score     {jaccard_score(orig_y_test, predictions, average='samples') * 100:02.1f}% (more is better)")
        logger.info(f"Exact match ratio {accuracy_score(orig_y_test, predictions) * 100:02.1f}% (more is better)")
        logger.info(f"Zero-one loss     {zero_one_loss(orig_y_test, predictions) * 100:.1f}% (less is better)\n")
        logger.info(f"Hamming loss      {hamming_loss(orig_y_test, predictions) * 100:.1f}% (less is better)")

        logger.info(f"F1 MACRO: {f1_score(orig_y_test, predictions, average='macro') * 100:.2}% (more is better)")
        logger.info(f"F1 MICRO: {f1_score(orig_y_test, predictions, average='micro') * 100:.2f}% (more is better)")
        logger.info(f"F1 SAMPLES: {f1_score(orig_y_test, predictions, average='samples') * 100:.2f}% (more is better)")

    logger.info("It is done")
