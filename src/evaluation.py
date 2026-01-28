import pandas as pd
from src.candidate_generation import compute_item_popularity, recommend_top_k

def time_based_split(
        interactions: pd.DataFrame,
        timestamp_col: str = 'last_interaction_ts',
        train_ratio: float = 0.8,
):
    interactions = interactions.sort_values(timestamp_col).reset_index(drop=True)
    split_index = int(len(interactions) * train_ratio)
    train = interactions.iloc[:split_index].copy()
    test = interactions.iloc[split_index:].copy()
    return train, test

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k

def recall_at_k(recommended: list, relevant: set) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended) & relevant) / len(relevant)

def evaluate_popularity_model(interactions: pd.DataFrame, k: int = 10):
    # 1. Split Data
    train, test = time_based_split(interactions)

    # 2. Compute Popularity
    item_popularity = compute_item_popularity(train)
    
    # --- OPTIMIZATION 1: Convert to List ---
    # Extract the item IDs into a standard Python list.
    # This list is already sorted by popularity from the compute function.
    top_popular_items = item_popularity["item_id"].tolist()
    # ---------------------------------------

    # --- OPTIMIZATION 2: Pre-compute Histories ---
    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history = test.groupby("user_id")["item_id"].apply(set).to_dict()

    precisions = []
    recalls = []
    
    # Iterate over unique test users
    test_users = list(test_history.keys())

    for user_id in test_users:
        relevant_items = test_history.get(user_id)
        if not relevant_items:
            continue

        seen_items = train_history.get(user_id, set())

        # Pass the LIST, not the dataframe
        recommended_items = recommend_top_k(
            user_id=user_id,
            user_history=seen_items,
            top_popular_items=top_popular_items,
            k=k,
        )

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items))

    return {
        "Precision@10": sum(precisions) / len(precisions) if precisions else 0.0,
        "Recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
        "num_test_users": len(precisions),
    }