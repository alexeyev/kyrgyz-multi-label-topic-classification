""" Will multilingual LLMs work?  """

import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, zero_one_loss, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer

logger = logging.getLogger("evaluation-neural")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("eval-neural.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t[%(levelname)s]\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def at_least_one_hit_rate(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray()

    hits_per_line = np.sum(y_true * y_pred, axis=1)
    return np.sum(hits_per_line > 0) / y_true.shape[0]


def multi_label_metrics(preds, labls, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))

    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # finally, compute metrics
    y_true = labls
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_sample_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    jaccard = jaccard_score(y_true=y_true, y_pred=y_pred, average='samples')
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    at_least_one = at_least_one_hit_rate(y_true=y_true, y_pred=y_pred)
    zero_one = zero_one_loss(y_true=y_true, y_pred=y_pred)
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

    # return as dictionary
    metrics_dict = {'f1_micro': f1_micro_average,
                    'f1_macro': f1_macro_average,
                    'f1_sample': f1_sample_average,
                    'jaccard': jaccard,
                    'accuracy': accuracy,
                    'at_least_one': at_least_one,
                    'zero_one_loss': zero_one,
                    'hamming_loss': hamming}
    return metrics_dict


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(preds=preds, labls=p.label_ids)
    return result


if __name__ == "__main__":
    np.random.seed(100)
    torch.manual_seed(0)

    # Reading data
    df_train = pd.read_csv("data/qad-train_1000.csv")
    df_test = pd.read_csv("data/qad-test_0500.csv")

    orig_train_text = np.array(df_train["text"].map(str))
    orig_test_text = np.array(df_test["text"].map(str))

    mlb = MultiLabelBinarizer(sparse_output=False)
    extract_labels = lambda d: d["labels_set"].map(lambda x: tuple(x.split(",")))
    orig_y_train = mlb.fit_transform(extract_labels(df_train))
    orig_y_test = mlb.transform(extract_labels(df_test))
    labels = mlb.classes_

    LABELS, LABELS_ORDERED = len(labels), labels
    label2id = {name: idx for idx, name in enumerate(LABELS_ORDERED)}
    id2label = {v: k for k, v in label2id.items()}

    # Splitting the data
    stratifier = IterativeStratification(n_splits=2, order=1)
    train_indexes, dev_indexes = next(stratifier.split(list(orig_train_text), orig_y_train))

    logger.info(f"{train_indexes.shape}, {dev_indexes.shape}")

    cv = [(train_indexes, dev_indexes), (dev_indexes, train_indexes)]
    train_text, y_train = orig_train_text[train_indexes], orig_y_train[train_indexes, :]
    dev_text, y_dev = orig_train_text[dev_indexes], orig_y_train[dev_indexes, :]

    # Setting up the LLM
    model_name, maxl = "xlm-roberta-large", 512
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=maxl)


    def process_data(example):
        """ Tokenizing and adding labels """
        text = example["text"]
        labls = example["labels"]
        tokenized = tokenizer(text,
                              max_length=maxl,
                              padding="max_length",
                              truncation=True)
        tokenized["labels"] = labls
        return tokenized


    # Preparing the datasets, conversion into the HuggingFace datasets format
    orig_train = Dataset.from_dict({"text": orig_train_text,
                                    "labels": orig_y_train.astype(np.float32)})
    train = Dataset.from_dict({"text": train_text,
                               "labels": y_train.astype(np.float32)})
    dev = Dataset.from_dict({"text": dev_text,
                             "labels": y_dev.astype(np.float32)})
    test = Dataset.from_dict({"text": orig_test_text,
                              "labels": orig_y_test.astype(np.float32)})

    encoded_orig_train = orig_train.map(process_data,
                                        batched=True,
                                        remove_columns=orig_train.column_names)
    encoded_train = train.map(process_data,
                              batched=True,
                              remove_columns=train.column_names)
    encoded_dev = dev.map(process_data,
                          batched=True,
                          remove_columns=dev.column_names)
    encoded_test = test.map(process_data,
                            batched=True,
                            remove_columns=test.column_names)

    encoded_orig_train.set_format("torch")
    encoded_train.set_format("torch")
    encoded_dev.set_format("torch")
    encoded_test.set_format("torch")

    # Setting up the metrics of interest
    # Loading the model, setting hyperparameters and training
    SecClass = AutoModelForSequenceClassification
    model = SecClass.from_pretrained(model_name,
                                     problem_type="multi_label_classification",
                                     num_labels=len(labels),
                                     id2label=id2label,
                                     label2id=label2id)

    model.to("cuda")

    args = TrainingArguments(
        "roberta-finetuned-kyrgyz-news-multilabel",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        load_best_model_at_end=True,
        metric_for_best_model="jaccard"
    )

    optimizer1 = AdamW(params=model.parameters(),
                       amsgrad=True,
                       weight_decay=0.01,
                       lr=2e-5)
    optimizer2 = torch.optim.ASGD(params=model.parameters(),
                                  weight_decay=0.01)

    scheduler1 = ExponentialLR(gamma=1.0, optimizer=optimizer1)
    scheduler2 = LinearLR(optimizer=optimizer1)

    trainer = Trainer(
        model,
        args,
        optimizers=(optimizer1, scheduler1),
        train_dataset=encoded_train,
        eval_dataset=encoded_dev,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluation on test data
    predictions, labels, metrics = trainer.predict(encoded_test)

    logger.info(f"{metrics}")
