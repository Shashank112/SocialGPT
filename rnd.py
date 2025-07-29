import streamlit as st
import pandas as pd
from src.config import DATA_PATH

from src.load_data import load_engagement_data
from src.preprocessing import apply_cleaning
from src.sentiment_analysis import get_sentiment_dataframe

from src.analytics import (
    plot_sentiment_distribution,
    plot_keyword_trends,
    golden_hour_analysis,
    plot_intent_distribution,
)
from src.intent import (
    cluster_comments,
    get_top_words_per_cluster,
    assign_intent_names,
)
from src.visualizations import plot_intent_tsne
from src.trend_analysis import discover_micro_influencers

st.set_page_config(page_title="Instagram Comment Insight Dashboard", layout="wide")

st.title("📊 Scrollmark Instagram Comment Insights")
st.markdown(
    "Analyze buyer sentiment, trends, intent clusters, and discover influencer signals."
)

# Load data
st.sidebar.header("Upload/Load Data")
df = load_engagement_data(DATA_PATH)

if df is not None:
    st.success("✅ Data loaded successfully!")
    st.write("Sample Data:", df.head())

    # Preprocessing
    df = apply_cleaning(df)
    df = get_sentiment_dataframe(df)

    st.header("1️⃣ Sentiment Analysis")
    st.pyplot(plot_sentiment_distribution(df))

    st.header("2️⃣ Intent Clustering & Visualization")
    df_clustered, labels = cluster_comments(df[df["sentiment"] == "positive"])
    df_clustered["sentiment"] = "positive"

    cluster_keywords = get_top_words_per_cluster(df_clustered)
    df_clustered = assign_intent_names(df_clustered, cluster_keywords)

    st.pyplot(plot_intent_tsne(df_clustered))
    st.pyplot(plot_intent_distribution(df_clustered))

    with st.expander("📌 Cluster Descriptions"):
        for cluster_id, name in cluster_keywords.items():
            st.markdown(f"**Cluster {cluster_id} – {name}**")

    st.header("3️⃣ Golden Hour Purchase Intent")
    golden_fig = golden_hour_analysis(df)
    st.pyplot(golden_fig)

    st.header("4️⃣ Micro-Influencer Discovery")
    influencer_df = discover_micro_influencers(df)
    st.dataframe(influencer_df)

    st.header("5️⃣ Keyword Trend Explorer")
    st.pyplot(plot_keyword_trends(df))

    st.markdown("---")
    st.info("Built by Probietech · Powered by AI ✨")
else:
    st.warning("Please make sure `engagements.csv` is available in the `data/` folder.")
