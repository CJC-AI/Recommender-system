import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, Set, Tuple


# -------------------------------------------------------
# Popularity (fallback)
# -------------------------------------------------------
def compute_item_popularity(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total interaction score per item.
    
    Used as fallback for cold-start scenarios.
    
    Args:
        train_df: Training interactions DataFrame
    
    Returns:
        DataFrame with item_id and interaction_score, sorted by popularity
    """
    item_popularity = (
        train_df
        .groupby("item_id")["interaction_score"]
        .sum()
        .reset_index()
        .sort_values("interaction_score", ascending=False)
    )
    return item_popularity


# -------------------------------------------------------
# Item–Item Co-occurrence Matrix
# -------------------------------------------------------
def build_cooccurrence_matrix(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    min_cooccurrence: int = 2,
) -> Dict[int, Dict[int, float]]:
    """
    Build item-item co-occurrence matrix with cosine normalization.
    
    For each user/session:
    1. Take all items interacted with
    2. Generate all item pairs
    3. Count co-occurrences across users
    4. Normalize using cosine similarity
    
    Args:
        interactions: DataFrame with user-item interactions
        user_col: Column name for user IDs
        item_col: Column name for item IDs
        min_cooccurrence: Minimum co-occurrences to include pair (filter noise)
    
    Returns:
        Nested dictionary: {item_id: {similar_item_id: similarity_score}}
    """
    # 1. Generate item pairs per user
    # Group items by user into sets for fast pair generation
    user_items = interactions.groupby(user_col)[item_col].apply(set).to_dict()
    
    # 2. Count co-occurrences
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    item_counts = defaultdict(int)
    
    for user_id, items in user_items.items():
        items_list = list(items)
        
        # Count individual item occurrences
        for item in items_list:
            item_counts[item] += 1
        
        # Generate all pairs and count co-occurrences
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                item_a = items_list[i]
                item_b = items_list[j]
                
                # Store both directions for symmetric access
                cooccurrence_counts[item_a][item_b] += 1
                cooccurrence_counts[item_b][item_a] += 1
    
    # 3. Normalize using cosine similarity 
    similarity_matrix = {}
    
    for item_a, cooccurrences in cooccurrence_counts.items():
        similarity_matrix[item_a] = {}
        
        for item_b, count in cooccurrences.items():
            # Filter out low co-occurrence pairs (noise reduction)
            if count < min_cooccurrence:
                continue
            
            # Cosine similarity: count / sqrt(count_a * count_b)
            denominator = np.sqrt(item_counts[item_a] * item_counts[item_b])
            
            if denominator > 0:
                similarity_matrix[item_a][item_b] = count / denominator
    
    return similarity_matrix


# -------------------------------------------------------
# Candidate Generation (Item → Item)
# -------------------------------------------------------
def get_similar_items(
    item_id: int,
    similarity_matrix: Dict[int, Dict[int, float]],
    top_k: int = 50,
) -> list:
    """
    Get top-K most similar items for a given item.
    
    Args:
        item_id: The item to find similar items for
        similarity_matrix: Pre-computed similarity matrix from build_cooccurrence_matrix
        top_k: Number of similar items to return
    
    Returns:
        List of similar item IDs, sorted by similarity (most similar first)
    """
    # Get similar items for this item
    similar_items_dict = similarity_matrix.get(item_id, {})
    
    # Sort by similarity score (descending) and take top K
    sorted_similar = sorted(
        similar_items_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract just the item IDs
    similar_items = [item for item, score in sorted_similar[:top_k]]
    
    return similar_items


# -------------------------------------------------------
# User-level Recommendation
# -------------------------------------------------------
def recommend_item_based(
    user_history: Set[int],
    similarity_matrix: Dict[int, Dict[int, float]],
    top_popular_items: list,
    k: int = 10,
    candidates_per_item: int = 50,
) -> list:
    """
    Recommend items using item-item collaborative filtering with explicit cold-start handling.
    
    ALGORITHM:
    1. For each item in user's history, find similar items
    2. Aggregate scores across all similar items
    3. Filter out items already in user history
    4. Return top-K by score
    
    COLD-START HANDLING:
    - If user has no history → recommend popular items
    - If items have no neighbors → fallback to popularity
    
    Args:
        user_history: Set of item IDs the user has interacted with
        similarity_matrix: Pre-computed item-item similarity matrix
        top_popular_items: Pre-sorted list of popular items (for cold-start)
        k: Number of recommendations to return
        candidates_per_item: Number of similar items to consider per history item
    
    Returns:
        List of recommended item IDs
    """
    # --- COLD-START 1: User has no history ---
    if not user_history:
        # Fallback to popularity-based recommendations
        return recommend_top_k_popular(
            user_history=set(),
            top_popular_items=top_popular_items,
            k=k
        )
    
    # Aggregate candidate items and their scores
    candidate_scores = {}
    items_with_neighbors = 0
    
    for item_id in user_history:
        # Get similar items from the matrix
        similar_items_dict = similarity_matrix.get(item_id, {})
        
        if similar_items_dict:
            items_with_neighbors += 1
            
            # Get top candidates for this item
            sorted_similar = sorted(
                similar_items_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:candidates_per_item]
            
            # Aggregate scores
            for candidate_item, similarity_score in sorted_similar:
                if candidate_item not in user_history:
                    candidate_scores[candidate_item] = (
                        candidate_scores.get(candidate_item, 0) + similarity_score
                    )
    
    # --- COLD-START 2: Items have no neighbors ---
    if items_with_neighbors == 0 or not candidate_scores:
        # None of the user's items had similar items in the matrix
        # Fallback to popularity
        return recommend_top_k_popular(
            user_history=user_history,
            top_popular_items=top_popular_items,
            k=k
        )
    
    # Sort candidates by aggregated score
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Extract top K item IDs
    recommendations = [item_id for item_id, score in sorted_candidates[:k]]
    
    # --- COLD-START 3: Not enough recommendations ---
    # If we don't have enough recommendations, fill with popular items
    if len(recommendations) < k:
        popular_backfill = recommend_top_k_popular(
            user_history=user_history.union(set(recommendations)),
            top_popular_items=top_popular_items,
            k=k - len(recommendations)
        )
        recommendations.extend(popular_backfill)
    
    return recommendations


# -------------------------------------------------------
# Popularity Recommendation
# -------------------------------------------------------
def recommend_top_k_popular(
    user_history: Set[int],
    top_popular_items: list,
    k: int = 10,
) -> list:
    """
    Recommend Top-K popular items the user has not interacted with yet.
    
    OPTIMIZATION:
    Instead of filtering a dataframe (slow), iterate through the
    pre-sorted list of popular items and pick the first k unseen ones.
    
    Args:
        user_history: Set of item IDs the user has already interacted with
        top_popular_items: Pre-sorted list of items by popularity (most popular first)
        k: Number of recommendations to return
    
    Returns:
        List of recommended item IDs
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