import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load local modules
from src.load_data import load_engagement_data
from src.preprocessing import apply_cleaning
from src.sentiment_analysis import get_sentiment_dataframe
from src.trend_analysis import compute_daily_sentiment_trend
from src.keyword_extraction import extract_top_ngrams
from src.intent import (
    cluster_comments,
    get_top_keywords_per_cluster,
    assign_intent_names,
)

from src.config import DATA_PATH

st.title("📊 Instagram Comment Analysis Dashboard")

# Load and prepare data
st.header("1. Data Loading & Preprocessing")
df = load_engagement_data(DATA_PATH)
df = apply_cleaning(df)
df = get_sentiment_dataframe(df)
st.success("Data loaded and processed successfully")

# Sentiment Distribution
st.header("2. Sentiment Distribution")
sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
st.pyplot(plt.gcf())
plt.clf()

# Daily sentiment trends
st.header("3. Sentiment Trends Over Time")
daily_percent = compute_daily_sentiment_trend(df)
for sentiment in ["positive", "neutral", "negative"]:
    plt.plot(
        daily_percent.index, daily_percent[sentiment], label=sentiment.capitalize()
    )

plt.title("Daily Sentiment Trend")
plt.xlabel("Date")
plt.ylabel("% of Comments")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()

# Top phrases
st.header("4. Top Positive and Negative Phrases")
pos_comments = df[df["sentiment"] == "positive"]["clean_text"].tolist()
neg_comments = df[df["sentiment"] == "negative"]["clean_text"].tolist()

top_pos_phrases = extract_top_ngrams(pos_comments)
top_neg_phrases = extract_top_ngrams(neg_comments)


def show_wordcloud(phrases, title):
    wc = WordCloud(width=800, height=400).generate_from_frequencies(dict(phrases))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.clf()


show_wordcloud(top_pos_phrases, "Positive Themes")
show_wordcloud(top_neg_phrases, "Negative Themes")

# Intent clustering
st.header("5. Comment Intent Clustering")
df_pos = df[df["sentiment"] == "positive"].copy()
labels, top_terms = cluster_comments(df_pos["clean_text"].tolist())
df_pos["intent_cluster"] = labels

# t-SNE projection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
X = vectorizer.fit_transform(df_pos["clean_text"].tolist())
X_dense = X.toarray()
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_dense)

df_pos["tsne_1"] = X_tsne[:, 0]
df_pos["tsne_2"] = X_tsne[:, 1]

df_clustered = df_pos.copy()
df_clustered["cluster"] = labels
cluster_names = get_top_keywords_per_cluster(df_clustered)
df_clustered = assign_intent_names(df_clustered, cluster_names)

sns.scatterplot(
    data=df_clustered, x="tsne_1", y="tsne_2", hue="cluster_name", palette="tab10"
)
plt.title("t-SNE Intent Clustering")
plt.legend(loc="upper right")
st.pyplot(plt.gcf())
plt.clf()

# Cluster bar chart
st.subheader("Top Intent Clusters")
cluster_counts = df_clustered["cluster_name"].value_counts()
sns.barplot(x=cluster_counts.values, y=cluster_counts.index)
plt.xlabel("Number of Comments")
plt.ylabel("Intent Cluster")
plt.title("Comment Clusters")
st.pyplot(plt.gcf())
plt.clf()

# Golden Hour Analysis
st.header("6. Golden Hour for Purchase Intent")
df["created_time"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["created_time"].dt.hour

purchase_keywords = [
    "buy",
    "ordered",
    "order",
    "purchase",
    "bought",
    "add to cart",
    "can’t wait",
    "trying this",
]


def is_purchase_intent(text):
    return any(word in text.lower() for word in purchase_keywords)


df["is_purchase_intent"] = df["clean_text"].apply(is_purchase_intent)

sns.countplot(data=df[df["is_purchase_intent"]], x="hour", palette="magma")
plt.title("Purchase Intent by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Comments")
st.pyplot(plt.gcf())
plt.clf()

st.success("📈 Dashboard Rendered Successfully")
