import pandas as pd
from src.candidate_generation import (
    compute_item_popularity,
    recommend_top_k_popular,
    recommend_item_based,
    build_cooccurrence_matrix,
)


# -------------------------------------------------------
# Time-Based split
# -------------------------------------------------------
def time_based_split(
    interactions: pd.DataFrame,
    timestamp_col: str = "last_interaction_ts",
    train_ratio: float = 0.8,
):
    """
    Split interactions by time.

    Train: first `train_ratio` fraction
    Test: remaining interactions
    
    Args:
        interactions: DataFrame with user-item interactions
        timestamp_col: Column name for timestamp
        train_ratio: Fraction of data to use for training (0.0 to 1.0)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    interactions = interactions.sort_values(timestamp_col).reset_index(drop=True)

    split_index = int(len(interactions) * train_ratio)

    train = interactions.iloc[:split_index].copy()
    test = interactions.iloc[split_index:].copy()

    return train, test



# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Calculate precision@K: fraction of top-K recommendations that are relevant.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider
    
    Returns:
        Precision score (0.0 to 1.0)
    """
    if not recommended:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set) -> float:
    """
    Calculate recall: fraction of relevant items that were recommended.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
    
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    return len(set(recommended) & relevant) / len(relevant)



# -------------------------------------------------------
# Evaluation Pipeline (Popularity)
# -------------------------------------------------------
def evaluate_popularity_model(interactions: pd.DataFrame, k: int = 10):
    """
    Evaluate popularity-based recommendation model.
    
    Args:
        interactions: Full interaction dataset
        k: Number of recommendations to generate per user
    
    Returns:
        Dictionary with evaluation metrics
    """
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
        recommended_items = recommend_top_k_popular(
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



# -------------------------------------------------------
# Evaluation Pipeline (Item-Based CF)
# -------------------------------------------------------
def evaluate_item_based_model(
    interactions: pd.DataFrame,
    k: int = 10,
    candidates_per_item: int = 50,
    min_cooccurrence: int = 2,
):
    """
    Evaluate item-based collaborative filtering model with explicit cold-start handling.
    
    PROCESS:
    1. Time-based train/test split
    2. Build co-occurrence matrix on training data
    3. Compute popularity fallback
    4. Generate recommendations for each test user
    5. Calculate precision and recall metrics
    
    Args:
        interactions: Full interaction dataset
        k: Number of recommendations to generate per user
        candidates_per_item: Number of similar items to consider per history item
        min_cooccurrence: Minimum co-occurrences to include in similarity matrix
    
    Returns:
        Dictionary with evaluation metrics and cold-start statistics
    """
    # 1. Split Data
    train, test = time_based_split(interactions)
    
    # 2. Build Co-occurrence Matrix
    print("Building co-occurrence matrix...")
    similarity_matrix = build_cooccurrence_matrix(
        interactions=train,
        min_cooccurrence=min_cooccurrence,
    )
    print(f"Matrix built: {len(similarity_matrix)} items with neighbors")
    
    # 3. Compute Popularity (for cold-start fallback)
    item_popularity = compute_item_popularity(train)
    top_popular_items = item_popularity["item_id"].tolist()
    
    # 4. Pre-compute Histories
    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history = test.groupby("user_id")["item_id"].apply(set).to_dict()
    
    # 5. Generate Recommendations and Evaluate
    precisions = []
    recalls = []
    cold_start_users = 0  # Users with no training history
    popularity_fallback_users = 0  # Users whose items had no neighbors
    
    test_users = list(test_history.keys())
    
    for user_id in test_users:
        relevant_items = test_history.get(user_id)
        if not relevant_items:
            continue
        
        seen_items = train_history.get(user_id, set())
        
        # Track cold-start scenarios
        if not seen_items:
            cold_start_users += 1
        
        # Generate recommendations using item-based CF
        recommended_items = recommend_item_based(
            user_history=seen_items,
            similarity_matrix=similarity_matrix,
            top_popular_items=top_popular_items,
            k=k,
            candidates_per_item=candidates_per_item,
        )
        
        # Calculate metrics
        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items))
    
    return {
        "Precision@10": sum(precisions) / len(precisions) if precisions else 0.0,
        "Recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
        "num_test_users": len(precisions),
        "cold_start_users": cold_start_users,
        "items_with_neighbors": len(similarity_matrix),
    }


# -------------------------------------------------------
# Comparison Pipeline
# -------------------------------------------------------
def compare_models(interactions: pd.DataFrame, k: int = 10):
    """
    Compare popularity-based and item-based collaborative filtering models.
    
    Args:
        interactions: Full interaction dataset
        k: Number of recommendations to generate per user
    
    Returns:
        Dictionary with comparison metrics
    """
    print("=" * 60)
    print("EVALUATING POPULARITY-BASED MODEL")
    print("=" * 60)
    popularity_metrics = evaluate_popularity_model(interactions, k=k)
    
    print("\nPopularity Model Results:")
    for metric, value in popularity_metrics.items():
        print(f"  {metric}: {value}")
    
    print("\n" + "=" * 60)
    print("EVALUATING ITEM-BASED COLLABORATIVE FILTERING MODEL")
    print("=" * 60)
    item_based_metrics = evaluate_item_based_model(interactions, k=k)
    
    print("\nItem-Based CF Model Results:")
    for metric, value in item_based_metrics.items():
        print(f"  {metric}: {value}")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Precision Improvement: {item_based_metrics['Precision@10'] - popularity_metrics['Precision@10']:+.4f}")
    print(f"Recall Improvement: {item_based_metrics['Recall@10'] - popularity_metrics['Recall@10']:+.4f}")
    
    return {
        "popularity": popularity_metrics,
        "item_based": item_based_metrics,
    }
