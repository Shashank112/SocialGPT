import re
import pandas as pd
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_comment(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text)  # normalize whitespace

    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(filtered)


def apply_cleaning(df: pd.DataFrame, column: str = "comment_text") -> pd.DataFrame:
    df[column] = df[column].astype(str)
    df["clean_text"] = df[column].apply(clean_comment)
    return df
