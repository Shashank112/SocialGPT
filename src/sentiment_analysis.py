import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> str:
    """Returns 'positive', 'negative', or 'neutral' sentiment"""
    if not text or not isinstance(text, str):
        return "neutral"

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def get_sentiment_dataframe(df) -> pd.DataFrame:
    df["sentiment"] = df["clean_text"].apply(analyze_sentiment)
    return df
