import pandas as pd

def compute_item_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total interaction score per item.
    """
    item_popularity = (
        train_df
        .groupby("item_id")["interaction_score"]
        .sum()
        .reset_index()
        .sort_values("interaction_score", ascending=False)
    )
    return item_popularity

def recommend_top_k(
        user_id: int, 
        train_df: pd.DataFrame,
        k: int = 10
) -> list:
    """
    Recommend Top-K popular items the user has not interacted with yet.
    """
    item_popularity = compute_item_popularity(train_df)

    # Get items the user has already interacted with
    user_items = train_df.loc[
        train_df["user_id"] == user_id, "item_id"
        ].unique()

    # Filter out items the user has already interacted with
    recommendations = item_popularity[
        ~item_popularity["item_id"].isin(user_items)
        ]
    
    return recommendations["item_id"].head(k).tolist()