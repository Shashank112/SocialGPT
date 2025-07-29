import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from collections import Counter

sns.set(style="whitegrid")


def plot_sentiment_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="sentiment", data=df, palette="pastel", ax=ax)
    ax.set_title("Sentiment Distribution")
    return fig


def plot_keyword_trends(df, time_col="timestamp", text_col="cleaned_comment"):
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["month"] = df[time_col].dt.to_period("M")

    # Basic keyword trend for most common word
    all_words = " ".join(df[text_col].dropna()).split()
    common_words = [word for word, count in Counter(all_words).most_common(5)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for word in common_words:
        df[word] = df[text_col].apply(
            lambda x: x.lower().split().count(word) if isinstance(x, str) else 0
        )
        df_grouped = df.groupby("month")[word].sum()
        df_grouped.index = df_grouped.index.astype(str)
        ax.plot(df_grouped.index, df_grouped.values, label=word)

    ax.set_title("Keyword Trends Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.xticks(rotation=45)
    return fig


def golden_hour_analysis(df, time_col="timestamp"):
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df[time_col].dt.hour

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="hour", data=df, palette="viridis", ax=ax)
    ax.set_title("Golden Hour Engagement")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Comments")
    return fig


def plot_intent_distribution(df_clustered, label_col="intent_name"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        y=label_col,
        data=df_clustered,
        palette="Set3",
        order=df_clustered[label_col].value_counts().index,
        ax=ax,
    )
    ax.set_title("Top Intent Clusters")
    ax.set_xlabel("Comment Count")
    ax.set_ylabel("Intent Cluster")
    return fig
