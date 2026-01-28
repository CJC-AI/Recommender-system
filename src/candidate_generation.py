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
        user_history: set, 
        top_popular_items: list, 
        k: int = 10
) -> list:
    """
    Recommend Top-K popular items the user has not interacted with yet.
    
    OPTIMIZATION:
    Instead of filtering a dataframe (slow), iterate through the
    pre-sorted list of popular items and pick the first k unseen ones.
    """
    recommendations = []
    
    for item in top_popular_items:
        # If user hasn't seen it, add it
        if item not in user_history:
            recommendations.append(item)
        
        # Stop once we have enough recommendations
        if len(recommendations) >= k:
            break
            
    return recommendations