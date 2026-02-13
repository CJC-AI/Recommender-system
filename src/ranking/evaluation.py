import pandas as pd
import numpy as np
import math
import os
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
    interactions = interactions.sort_values(timestamp_col).reset_index(drop=True)
    split_index  = int(len(interactions) * train_ratio)
    train = interactions.iloc[:split_index].copy()
    test  = interactions.iloc[split_index:].copy()
    return train, test


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended: return 0.0
    return len(set(recommended[:k]) & relevant) / k

def recall_at_k(recommended: list, relevant: set) -> float:
    if not relevant: return 0.0
    return len(set(recommended) & relevant) / len(relevant)

def _dcg(relevances: list) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    ranked_rel = [1 if item in relevant else 0 for item in recommended[:k]]
    dcg  = _dcg(ranked_rel)
    n_ideal = min(len(relevant), k)
    idcg    = _dcg([1] * n_ideal)
    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------------
# Evaluation: Popularity
# -------------------------------------------------------
def evaluate_popularity_model(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> dict:
    item_popularity   = compute_item_popularity(train)
    top_popular_items = item_popularity["item_id"].tolist()
    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history  = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    for user_id, relevant in test_history.items():
        if not relevant: continue
        seen = train_history.get(user_id, set())
        recs = recommend_top_k_popular(seen, top_popular_items, k=k)
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
    print("  Building co-occurrence matrix …")
    similarity_matrix = build_cooccurrence_matrix(train, min_cooccurrence=min_cooccurrence)
    print(f"  {len(similarity_matrix):,} items with neighbours")

    item_popularity   = compute_item_popularity(train)
    top_popular_items = item_popularity["item_id"].tolist()
    train_history = train.groupby("user_id")["item_id"].apply(set).to_dict()
    test_history  = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    cold_start_users = 0
    for user_id, relevant in test_history.items():
        if not relevant: continue
        seen = train_history.get(user_id, set())
        if not seen: cold_start_users += 1
        
        recs = recommend_item_based(
            seen, similarity_matrix, top_popular_items, k=k, candidates_per_item=candidates_per_item
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
    model_path: str = "artifacts/models/lr_ranker.joblib",
    k: int = 10,
    n_candidates: int = 50,
) -> dict:
    from src.ranking.infer import RankingContext, load_model, infer

    if not os.path.exists(model_path):
        print(f"  [WARNING] Model not found at {model_path}. Skipping ranker evaluation.")
        return {"Recall@10": 0.0, "NDCG@10": 0.0, "num_test_users": 0}

    print("  Building RankingContext …")
    ctx   = RankingContext(train)
    model = load_model(model_path)
    test_history = test.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    test_users     = list(test_history.keys())

    print(f"  Scoring {len(test_users):,} users...")
    for i, user_id in enumerate(test_users):
        relevant = test_history[user_id]
        if not relevant: continue

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
    model_path: str = "artifacts/models/lr_ranker.joblib",
    k: int = 10,
) -> dict:
    train, test = time_based_split(interactions)

    print("\n" + "=" * 60)
    print(" POPULARITY BASELINE")
    print("=" * 60)
    pop_metrics = evaluate_popularity_model(train, test, k=k)

    print("\n" + "=" * 60)
    print(" ITEM-ITEM CF")
    print("=" * 60)
    cf_metrics = evaluate_item_based_model(train, test, k=k)

    print("\n" + "=" * 60)
    print(" ITEM-ITEM CF  +  LEARNED RANKER")
    print("=" * 60)
    ranked_metrics = evaluate_ranked_model(train, test, model_path=model_path, k=k)

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