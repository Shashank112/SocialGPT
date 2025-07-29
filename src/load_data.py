# src/load_data.py
import pandas as pd
from datetime import datetime


def load_engagement_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean engagement data from CSV.
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace in column names
    df.columns = df.columns.str.strip()

    # Standardize timestamp
    def parse_timestamp(x):
        try:
            return pd.to_datetime(x, errors="coerce", utc=True)
        except:
            return pd.NaT

    df["timestamp"] = df["timestamp"].apply(parse_timestamp)

    # Drop rows with no comment text
    df = df[df["comment_text"].notnull()].copy()

    return df


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['comment_text'] = df['comment_text'].fillna('').str.strip()
        df['media_caption'] = df['media_caption'].fillna('').str.strip()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
