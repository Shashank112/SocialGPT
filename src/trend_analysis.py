import pandas as pd
# src/trends.py


def discover_micro_influencers(df: pd.DataFrame, min_comments: int = 5, min_followers: int = 500, max_followers: int = 10000):
    """
    Discover potential micro-influencers based on number of comments and follower count.
    
    Args:
        df (pd.DataFrame): DataFrame containing comment data with 'username' and 'followers' columns.
        min_comments (int): Minimum number of comments by the user.
        min_followers (int): Minimum followers to be considered a micro-influencer.
        max_followers (int): Maximum followers to be considered a micro-influencer.
    
    Returns:
        pd.DataFrame: Filtered micro-influencer candidates.
    """
    # Ensure 'username' and 'followers' exist
    if 'username' not in df.columns or 'followers' not in df.columns:
        raise ValueError("DataFrame must contain 'username' and 'followers' columns")

    # Count number of comments per user
    user_comments = df['username'].value_counts().reset_index()
    user_comments.columns = ['username', 'comment_count']

    # Get unique followers per user
    user_followers = df[['username', 'followers']].drop_duplicates()

    # Merge
    user_stats = pd.merge(user_comments, user_followers, on='username')

    # Filter based on thresholds
    influencers = user_stats[
        (user_stats['comment_count'] >= min_comments) &
        (user_stats['followers'] >= min_followers) &
        (user_stats['followers'] <= max_followers)
    ].sort_values(by='comment_count', ascending=False)

    return influencers

def compute_daily_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with daily sentiment percentages:
    positive, neutral, negative (normalized)
    """
    if 'timestamp' not in df or 'sentiment' not in df:
        raise ValueError("DataFrame must contain 'timestamp' and 'sentiment' columns")

    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    # Count of each sentiment per day
    daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    daily_sentiment['total'] = daily_sentiment.sum(axis=1)

    # Convert to percentage
    daily_percent = daily_sentiment[['positive', 'neutral', 'negative']].div(daily_sentiment['total'], axis=0) * 100

    return daily_percent
