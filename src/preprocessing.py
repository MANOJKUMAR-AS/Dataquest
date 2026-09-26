import re
import pandas as pd


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(text):
    text = normalize_text(text)

    # Keep words and simple apostrophes
    tokens = re.findall(r"\b[\w']+\b", text)

    return tokens


def prepare_dataframe(df):
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("").astype(str)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamps could not be parsed."
        )

    df["normalized_text"] = df["text"].apply(
        normalize_text
    )

    df["tokens"] = df["normalized_text"].apply(
        tokenize
    )

    return df