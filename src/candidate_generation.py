from pathlib import Path
import pandas as pd

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERACTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.csv"


def load_interactions() -> pd.DataFrame:
    """
    Load the processed interactions dataset.
    """
    return pd.read_csv(INTERACTIONS_PATH)

def compute_item_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total interaction score per item.
    """
    item_popularity = (
        interactions
        .groupby("item_id")["interaction_score"]
        .sum()
        .reset_index()
        .sort_values("interaction_score", ascending=False)
    )
    return item_popularity

def recommend_top_k(user_id: int, k: int = 10) -> pd.DataFrame:
    """
    Recommend Top-K popular items the user has not interacted with yet.
    """
    interactions = load_interactions()
    item_popularity = compute_item_popularity(interactions)

    # Get items the user has already interacted with
    user_items = interactions.loc[
        interactions["user_id"] == user_id, "item_id"
        ].unique()

    # Filter out items the user has already interacted with
    recommendations = item_popularity[
        ~item_popularity["item_id"].isin(user_items)
        ]
    
    return recommendations.head(k)