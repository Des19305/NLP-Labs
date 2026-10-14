"""Reusable preprocessing helpers for LAB 04: Text Classification."""

import re
import pandas as pd


def load_sms_spam_dataset(path):
    """
    Load the SMS Spam Collection TSV file.

    Expected raw format: two tab-separated columns:
      v1 = label (ham/spam)
      v2 = message text

    Also supports a CSV with columns named label/text or v1/v2.
    """
    path = str(path)

    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
        # If this looks like a headered CSV or a one-column parse, try normal CSV loading.
        if df.shape[1] != 2 or (len(df) > 0 and str(df.iloc[0]["label"]).strip().lower() in {"label", "v1"}):
            raise ValueError("Retrying as a headered CSV.")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")

    df.columns = [str(column).strip().lower() for column in df.columns]

    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    elif {"label", "text"}.issubset(df.columns):
        pass
    elif len(df.columns) >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "text"]
    else:
        raise ValueError(
            "Could not identify label and text columns. Expected v1/v2 or label/text."
        )

    df = df[["label", "text"]].copy()
    df["label"] = df["label"].astype("string").str.strip().str.lower()
    df["text"] = df["text"].astype("string")
    df = df.replace({"<NA>": pd.NA})
    return df


def basic_clean_text(text):
    """Lightweight text normalization; keeps words and numbers."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df):
    """Return a copy with normalized text and whitespace-trimmed labels."""
    result = df.copy()
    result["label"] = result["label"].astype("string").str.strip().str.lower()
    result["text"] = result["text"].map(basic_clean_text)
    return result
