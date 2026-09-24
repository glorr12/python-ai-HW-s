
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from prepare_data import LABELS

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_NAME = os.getenv("BASE_MODEL", "distilbert-base-uncased")
OUTPUT_DIR = HERE / "model"

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}


def load_splits():
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro")}


def run_baseline(train, test) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vectorizer.fit_transform(train["text"]), train["label"])
    pred = clf.predict(vectorizer.transform(test["text"]))
    return metrics(test["label"], pred)


def to_dataset(df, tokenizer) -> Dataset:
    ds = Dataset.from_dict({"text": df["text"].tolist(),
                            "labels": [label2id[l] for l in df["label"]]})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)


def finetune(train, test, model_name=MODEL_NAME, output_dir=OUTPUT_DIR, epochs=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train, tokenizer),
        eval_dataset=to_dataset(test, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: metrics(p.label_ids, np.argmax(p.predictions, axis=-1)),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def evaluate(trainer, test) -> dict:
    output = trainer.predict(to_dataset(test, trainer.processing_class))
    pred = [id2label[i] for i in np.argmax(output.predictions, axis=-1)]
    true = test["label"].tolist()

    print(classification_report(true, pred, labels=LABELS, zero_division=0))
    print("Матрица ошибок (строки — истина, столбцы — прогноз):", LABELS)
    print(confusion_matrix(true, pred, labels=LABELS))

    errors = [(t, p, x) for t, p, x in zip(true, pred, test["text"]) if t != p]
    if errors:
        print("\nОшибки модели:")
        for t, p, x in errors:
            print(f"  [{t} -> {p}] {x}")
    return metrics(true, pred)


def predict(texts, model_dir=OUTPUT_DIR):
    from transformers import pipeline
    classifier = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir))
    return [r["label"] for r in classifier([t.lower() for t in texts])]


def main():
    train, test = load_splits()
    print(f"Train: {len(train)}, Test: {len(test)}\n")

    baseline = run_baseline(train, test)
    print(f"Baseline TF-IDF + LogReg: {baseline}\n")

    trainer = finetune(train, test)
    result = evaluate(trainer, test)

    print(f"\nИтог на test:\n  baseline   {baseline}\n  DistilBERT {result}")

    examples = ["I can't sign in to my profile",
                "The host charged my card two times",
                "Please help me move my stay to August"]
    for text, label in zip(examples, predict(examples)):
        print(f"  {label:8} <- {text}")


if __name__ == "__main__":
    main()
