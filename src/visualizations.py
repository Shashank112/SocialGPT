# src/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_intent_tsne(df_clustered, tsne_x='tsne_1', tsne_y='tsne_2', label_col='intent_name'):
    """
    Visualizes intent clusters using t-SNE reduced dimensions.

    Args:
        df_clustered (pd.DataFrame): DataFrame with t-SNE coordinates and intent labels.
        tsne_x (str): Column name for x-axis of t-SNE.
        tsne_y (str): Column name for y-axis of t-SNE.
        label_col (str): Column with cluster names (intent labels).
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_clustered[tsne_x],
        y=df_clustered[tsne_y],
        hue=df_clustered[label_col],
        palette='Set2',
        s=60,
        alpha=0.8,
        edgecolor='k'
    )
    plt.title("t-SNE Visualization of Buyer Intents", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Intent", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
