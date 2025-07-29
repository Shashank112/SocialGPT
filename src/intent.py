import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Tuple
from collections import Counter
import numpy as np

def cluster_comments(
    comments: List[str], num_clusters: int = 5
) -> Tuple[List[int], List[str]]:
    """
    Perform KMeans clustering on cleaned comment text.
    Returns:
        cluster_labels: Cluster number per comment
        top_terms_per_cluster: Top keywords describing each cluster
    """
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
    X = vectorizer.fit_transform(comments)

    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    labels = model.labels_

    # Get top terms per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    top_terms = [
        ", ".join([terms[ind] for ind in order_centroids[i, :5]])
        for i in range(num_clusters)
    ]

    return labels, top_terms


def get_top_words_per_cluster(
    df, cluster_col="cluster", text_col="clean_comment", top_n=10
):
    for cluster in sorted(df[cluster_col].unique()):
        cluster_comments = df[df[cluster_col] == cluster][text_col].tolist()
        all_words = " ".join(cluster_comments).split()
        top_words = Counter(all_words).most_common(top_n)
        print(f"\nCluster {cluster} Top Words:")
        for word, count in top_words:
            print(f"{word}: {count}")


def get_top_keywords_per_cluster(df: pd.DataFrame, n_terms: int = 5) -> dict:
    """
    Returns a dictionary of cluster_id -> top keywords from clean_text.
    """
    cluster_names = {}
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_texts = df[df["cluster"] == cluster_id]["clean_text"]
        
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, max_features=1000)
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words="english")

        X = vectorizer.fit_transform(cluster_texts)
        
        tfidf_sum = X.sum(axis=0).A1
        keywords = np.array(vectorizer.get_feature_names_out())[np.argsort(tfidf_sum)[::-1][:n_terms]]
        cluster_names[cluster_id] = ", ".join(keywords)

    return cluster_names


def assign_intent_names(df: pd.DataFrame, cluster_names: dict) -> pd.DataFrame:
    df["cluster_name"] = df["cluster"].map(cluster_names)
    return df