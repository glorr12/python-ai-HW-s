
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
LABELS = ["account", "booking", "listing", "payment", "review"]

TYPO_FIXES = {
    r"resrvation": "reservation", r"paymnet": "payment", r"pasword": "password",
    r"lisitng": "listing", r"reveiw": "review",
    r"cancelll": "cancell", r"\bcancell\b": "cancel",
}
MIN_WORDS = 3


def clean_text(text) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    for bad, good in TYPO_FIXES.items():
        text = re.sub(bad, good, text)
    return text


def clean_label(label) -> str:
    return "" if pd.isna(label) else str(label).strip().lower()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": len(df)}
    df = df.assign(text=df["text"].map(clean_text), label=df["label"].map(clean_label))

    df = df[df["label"].isin(LABELS)]
    report["after_label_filter"] = len(df)

    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    report["after_short_filter"] = len(df)

    df = df.drop_duplicates(subset="text")
    report["after_dedup"] = len(df)
    return df.reset_index(drop=True), report


def split(df: pd.DataFrame, test_size=0.2, seed=42):
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


def main():
    raw = pd.read_csv(DATA_DIR / "raw_tickets.csv", keep_default_na=False)
    clean, report = clean_dataframe(raw)
    train, test = split(clean)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print("Очистка:", report)
    print(f"Train: {len(train)}, Test: {len(test)}")
    print("Классы в train:", train["label"].value_counts().to_dict())
    return train, test


if __name__ == "__main__":
    main()
