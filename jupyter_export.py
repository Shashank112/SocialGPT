# %%
import sys
import os
import pandas as pd

# Add the parent directory of `src` to the path
sys.path.append(os.path.abspath('../')) 

# %%
from src.load_data import load_engagement_data
from src.config import DATA_PATH
print(DATA_PATH)
print("Looking for file at:", DATA_PATH)
print("File exists?", os.path.exists(DATA_PATH))
df = load_engagement_data(DATA_PATH)
df.head()
df.info()
df['timestamp'].describe()

# %%
# import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# %%
# Check distribution of sentiment labels
from src.preprocessing import apply_cleaning
df = apply_cleaning(df)
df.head()
# df[['comment_text', 'clean_text']].head(10)

# %%
from src.sentiment_analysis import get_sentiment_dataframe

df = get_sentiment_dataframe(df)
print(df.head())

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
plt.title("Sentiment Distribution of Comments")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# %%
print(df[['comment_text', 'sentiment']].sample(10))

# %%
from src.trend_analysis import compute_daily_sentiment_trend
daily_percent = compute_daily_sentiment_trend(df)
daily_percent.head()

# %%
plt.figure(figsize=(12, 6))
for sentiment in ['positive', 'neutral', 'negative']:
    plt.plot(daily_percent.index, daily_percent[sentiment], label=sentiment.capitalize())

plt.title("Daily Sentiment Trend of Instagram Comments (March 2025)")
plt.xlabel("Date")
plt.ylabel("Sentiment %")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Show spike days
high_neg = daily_percent[daily_percent['negative'] > 10]
high_pos = daily_percent[daily_percent['positive'] > 40]

print("High negativity days:\n", high_neg)
print("High positivity days:\n", high_pos)

# %%
from src.keyword_extraction import extract_top_ngrams
# 📌 Separate Clean Text by Sentiment
pos_comments = df[df['sentiment'] == 'positive']['clean_text'].tolist()
neg_comments = df[df['sentiment'] == 'negative']['clean_text'].tolist()

# 📌 Extract Top Phrases
top_pos_phrases = extract_top_ngrams(pos_comments, ngram_range=(1,2), top_k=20)
top_neg_phrases = extract_top_ngrams(neg_comments, ngram_range=(1,2), top_k=20)

# 📌 Print Results
print("🔹 Top Positive Phrases")
for phrase, count in top_pos_phrases:
    print(f"{phrase}: {count}")

print("\n🔻 Top Negative Phrases")
for phrase, count in top_neg_phrases:
    print(f"{phrase}: {count}")

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(phrases, title="Word Cloud"):
    word_freq = dict(phrases)
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate
show_wordcloud(top_pos_phrases, title="Positive Comment Themes")
show_wordcloud(top_neg_phrases, title="Negative Comment Themes")

# %%
from src.intent import cluster_comments

# 📌 Optionally filter only positive comments
df_pos = df[df['sentiment'] == "positive"].copy()

# 📌 Run clustering
labels, top_terms = cluster_comments(df_pos["clean_text"].tolist(), num_clusters=5)

# 📌 Add back to DataFrame
df_pos["intent_cluster"] = labels

# 📌 Show summary
for i, terms in enumerate(top_terms):
    print(f"\n🔹 Cluster {i}: {terms}")
    print(df_pos[df_pos["intent_cluster"] == i]["comment_text"].sample(3, random_state=42).tolist())

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
# Cluster comments and get vector representation
comments = df_pos["clean_text"].tolist()

# Same vectorizer as in src/intent.py
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
X = vectorizer.fit_transform(comments)

df_pos["intent_cluster"] = labels

# Convert sparse TF-IDF to dense for t-SNE
X_dense = X.toarray()

# Run t-SNE
tsne = tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
X_tsne = tsne.fit_transform(X_dense)

# Add to DataFrame
df_pos["tsne_1"] = X_tsne[:, 0]
df_pos["tsne_2"] = X_tsne[:, 1]

# %%
#Step 5: Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="intent_cluster",
    palette="tab10",
    data=df_pos,
    alpha=0.7
)

plt.title("💬 Comment Intent Clusters via t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# %%

# from src.intent import get_top_words_per_cluster
df_clustered = df_pos.copy()  # or df_sentiment if you're working from original
df_clustered["cluster"] = labels
df_clustered["x"] = X_tsne[:, 0]
df_clustered["y"] = X_tsne[:, 1]


# get_top_words_per_cluster(df_clustered, cluster_col='cluster', text_col='clean_comment')


# %%
from src.intent import get_top_keywords_per_cluster, assign_intent_names

# Get readable names
cluster_names = get_top_keywords_per_cluster(df_clustered)
df_clustered = assign_intent_names(df_clustered, cluster_names)

# Preview
df_clustered[["cluster", "cluster_name"]].drop_duplicates()


# %%
# Count the number of comments per cluster name
cluster_counts = df_clustered["cluster_name"].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.values, y=cluster_counts.index, palette="viridis")

plt.title("Top Clusters by Comment Count")
plt.xlabel("Number of Comments")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()


# %%
#Sentiment vs Hour Chart
df['created_time'] = pd.to_datetime(df['timestamp'])  # adjust if column name differs
df['hour'] = df['created_time'].dt.hour
df['day'] = df['created_time'].dt.day_name()

# Plot sentiment frequency by hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='hour', hue='sentiment', palette='Set2')
plt.title('Sentiment vs Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Comments')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()


# %%
#Step 4: Purchase Intent Keyword vs Hour

purchase_keywords = ['buy', 'ordered', 'order', 'purchase', 'bought', 'add to cart', 'can’t wait', 'trying this']

def is_purchase_intent(text):
    return any(word in text.lower() for word in purchase_keywords)

df['is_purchase_intent'] = df['clean_text'].apply(is_purchase_intent)

# Plot purchase intent by hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['is_purchase_intent']], x='hour', palette='magma')
plt.title('Purchase Intent Comments by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('High Intent Comments')
plt.tight_layout()
plt.show()

# %%
# Step 5: Identify the Golden Hour(s)

df['hour'] = df['timestamp'].dt.hour
df['is_purchase_intent'] = df['clean_text'].apply(is_purchase_intent)

# Top hours by positive sentiment
golden_sentiment = df[df['sentiment'] == 'positive']['hour'].value_counts().sort_index()

# Top hours by purchase intent
golden_purchase = df[df['is_purchase_intent']]['hour'].value_counts().sort_index()


plt.figure(figsize=(14, 6))
sns.lineplot(x=golden_sentiment.index, y=golden_sentiment.values, label='Positive Sentiment', marker='o')
sns.lineplot(x=golden_purchase.index, y=golden_purchase.values, label='Purchase Intent', marker='o')
plt.title('Golden Hour: Positive Sentiment vs Purchase Intent by Hour')
plt.xlabel('Hour of Day (0–23)')
plt.ylabel('Number of Comments')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
summary = pd.DataFrame({
    'hour': range(24),
    'positive_sentiment': golden_sentiment.reindex(range(24), fill_value=0).values,
    'purchase_intent': golden_purchase.reindex(range(24), fill_value=0).values
})
summary.to_csv('output/golden_hour_summary.csv', index=False)

# %%



