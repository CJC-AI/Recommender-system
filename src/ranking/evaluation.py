import pandas as pd
import numpy as np
import math
from typing import Dict, Set

from src.candidate_generation import (
    compute_item_popularity,
    recommend_top_k_popular,
    recommend_item_based,
    build_cooccurrence_matrix,
)


# -------------------------------------------------------
# Time-Based Split
# -------------------------------------------------------
def time_based_split(
    interactions: pd.DataFrame,
    timestamp_col: str = "last_interaction_ts",
    train_ratio: float = 0.8,
):
    """
    Split interactions by time.

    Train: first `train_ratio` fraction
    Test:  remaining interactions

    Args:
        interactions: DataFrame with user-item interactions
        timestamp_col: Column name for timestamp
        train_ratio:   Fraction of data to use for training

    Returns:
        (train_df, test_df)
    """
    interactions = interactions.sort_values(timestamp_col).reset_index(drop=True)
    split_index  = int(len(interactions) * train_ratio)
    train = interactions.iloc[:split_index].copy()
    test  = interactions.iloc[split_index:].copy()
    return train, test


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if not recommended:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set) -> float:
    """Fraction of relevant items that appear in the recommendation list."""
    if not relevant:
        return 0.0
    return len(set(recommended) & relevant) / len(relevant)


def _dcg(relevances: list) -> float:
    """DCG for a ranked list of binary relevances (0/1)."""
    return sum(
        rel / math.log2(i + 2)          # i is 0-based  ->  denominator = log2(rank + 1)
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    NDCG@K for a single user.

    Args:
        recommended: Ranked list of item_ids (most confident first)
        relevant:    Ground-truth set of item_ids the user interacted with in test
        k:           Cut-off rank

    Returns:
        NDCG@K score (0.0 - 1.0)
    """
    ranked_rel = [1 if item in relevant else 0 for item in recommended[:k]]

    dcg  = _dcg(ranked_rel)

    # Ideal: all relevant items first, up to k
    n_ideal = min(len(relevant), k)
    idcg    = _dcg([1] * n_ideal)

    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------------
# Evaluation: Popularity
# -------------------------------------------------------
def evaluate_popularity_model(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Evaluate popularity-based recommendations on the test set.

    Args:
        train: Training interactions
        test:  Test interactions
        k:     Number of recommendations

    Returns:
        {"Recall@10": ..., "NDCG@10": ..., "num_test_users": ...}
    """
    item_popularity   = compute_item_popularity(train)
    top_popular_items = item_popularity["item_id"].tolist()

    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history  = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []

    for user_id, relevant in test_history.items():
        if not relevant:
            continue
        seen = train_history.get(user_id, set())
        recs = recommend_top_k_popular(
            user_history=seen,
            top_popular_items=top_popular_items,
            k=k,
        )
        recalls.append(recall_at_k(recs, relevant))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        "Recall@10":      np.mean(recalls) if recalls else 0.0,
        "NDCG@10":        np.mean(ndcgs)   if ndcgs   else 0.0,
        "num_test_users": len(recalls),
    }


# -------------------------------------------------------
# Evaluation: Item-Item CF
# -------------------------------------------------------
def evaluate_item_based_model(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    k: int = 10,
    candidates_per_item: int = 50,
    min_cooccurrence: int = 2,
) -> dict:
    """
    Evaluate item-based CF on the test set.

    Args:
        train:               Training interactions
        test:                Test interactions
        k:                   Number of recommendations
        candidates_per_item: Neighbours considered per history item
        min_cooccurrence:    Threshold for co-occurrence matrix

    Returns:
        {"Recall@10": ..., "NDCG@10": ..., "num_test_users": ...,
         "cold_start_users": ..., "items_with_neighbors": ...}
    """
    print("  Building co-occurrence matrix …")
    similarity_matrix = build_cooccurrence_matrix(
        interactions=train,
        min_cooccurrence=min_cooccurrence,
    )
    print(f"  {len(similarity_matrix):,} items with neighbours")

    item_popularity   = compute_item_popularity(train)
    top_popular_items = item_popularity["item_id"].tolist()

    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history  = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    cold_start_users = 0

    for user_id, relevant in test_history.items():
        if not relevant:
            continue
        seen = train_history.get(user_id, set())
        if not seen:
            cold_start_users += 1

        recs = recommend_item_based(
            user_history=seen,
            similarity_matrix=similarity_matrix,
            top_popular_items=top_popular_items,
            k=k,
            candidates_per_item=candidates_per_item,
        )
        recalls.append(recall_at_k(recs, relevant))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        "Recall@10":            np.mean(recalls) if recalls else 0.0,
        "NDCG@10":              np.mean(ndcgs)   if ndcgs   else 0.0,
        "num_test_users":       len(recalls),
        "cold_start_users":     cold_start_users,
        "items_with_neighbors": len(similarity_matrix),
    }


# -------------------------------------------------------
# Evaluation: Item-Item CF  +  Learned Ranker
# -------------------------------------------------------
def evaluate_ranked_model(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    model_path: str = "models/xgb_ranker.json",
    k: int = 10,
    n_candidates: int = 50,
) -> dict:
    """
    End-to-end evaluation: candidate generation -> feature computation ->
    model scoring -> top-K.  Uses infer.py so the feature + scoring logic
    lives in exactly one place.

    Imports are lazy so this module loads even when no model exists yet.

    Args:
        train:        Training interactions
        test:         Test interactions
        model_path:   Path to trained model (.joblib or .json)
        k:            Final recommendations returned per user
        n_candidates: Candidate pool size before re-ranking

    Returns:
        {"Recall@10": ..., "NDCG@10": ..., "num_test_users": ...}
    """
    from src.infer import RankingContext, load_model, infer

    print("  Building RankingContext …")
    ctx   = RankingContext(train)
    model = load_model(model_path)

    test_history = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    test_users     = list(test_history.keys())

    for i, user_id in enumerate(test_users):
        relevant = test_history[user_id]
        if not relevant:
            continue

        recs = infer(user_id, ctx, model, n_candidates=n_candidates, k=k)

        recalls.append(recall_at_k(recs, relevant))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

        if (i + 1) % 1000 == 0:
            print(f"    scored {i + 1:,}/{len(test_users):,} users …")

    return {
        "Recall@10":      np.mean(recalls) if recalls else 0.0,
        "NDCG@10":        np.mean(ndcgs)   if ndcgs   else 0.0,
        "num_test_users": len(recalls),
    }


# -------------------------------------------------------
# Side-by-side comparison
# -------------------------------------------------------
def compare_models(
    interactions: pd.DataFrame,
    model_path: str = "models/xgb_ranker.json",
    k: int = 10,
) -> dict:
    """
    Run all three systems on the same train/test split and print an
    aligned comparison table.

    Args:
        interactions: Full interaction dataset
        model_path:   Path to the trained ranker
        k:            Recommendation cut-off

    Returns:
        {"popularity": {…}, "item_based": {…}, "ranked": {…}}
    """
    # Single split shared by all three systems
    train, test = time_based_split(interactions)

    # ----- Popularity -----
    print("\n" + "=" * 60)
    print(" POPULARITY BASELINE")
    print("=" * 60)
    pop_metrics = evaluate_popularity_model(train, test, k=k)

    # ----- Item-Item CF -----
    print("\n" + "=" * 60)
    print(" ITEM-ITEM CF")
    print("=" * 60)
    cf_metrics = evaluate_item_based_model(train, test, k=k)

    # ----- Item-Item CF + Learned Ranker -----
    print("\n" + "=" * 60)
    print(" ITEM-ITEM CF  +  LEARNED RANKER")
    print("=" * 60)
    ranked_metrics = evaluate_ranked_model(
        train, test, model_path=model_path, k=k
    )

    # ----- Aligned table -----
    print("\n" + "=" * 60)
    print(" COMPARISON")
    print("=" * 60)
    header = f"{'Model':<30} {'Recall@10':>12} {'NDCG@10':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'Popularity':<30} {pop_metrics['Recall@10']:>12.4f} {pop_metrics['NDCG@10']:>12.4f}")
    print(f"{'Item-Item CF':<30} {cf_metrics['Recall@10']:>12.4f} {cf_metrics['NDCG@10']:>12.4f}")
    print(f"{'Item-Item CF + Ranker':<30} {ranked_metrics['Recall@10']:>12.4f} {ranked_metrics['NDCG@10']:>12.4f}")
    print("=" * 60)

    return {
        "popularity": pop_metrics,
        "item_based": cf_metrics,
        "ranked":     ranked_metrics,
    }