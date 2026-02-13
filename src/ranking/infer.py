import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
from typing import Dict, Set

from src.candidate_generation import (
    build_cooccurrence_matrix,
    compute_item_popularity,
    recommend_item_based,
)
from src.ranking.dataset import fast_compute_features


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERACTIONS_PATH   = os.path.join(BASE_DIR, "data", "processed", "interactions.csv")
LR_MODEL_PATH       = os.path.join(BASE_DIR, "artifacts", "models", "lr_ranker.joblib")
XGB_MODEL_PATH      = os.path.join(BASE_DIR, "artifacts", "models", "xgb_ranker.json")

# Matches the 5-feature schema of your trained model
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "item_interaction_count",       
    "user_history_count",           
    "time_since_last_interaction",  
]

N_CANDIDATES = 50      # candidates pulled from item-item CF
K_FINAL      = 10      # final recommendations returned


# -------------------------------------------------------
# Shared State 
# -------------------------------------------------------
class RankingContext:
    """Holds pre-computed artifacts for scoring."""

    def __init__(self, train_df: pd.DataFrame, min_cooccurrence: int = 2):
        print("Building RankingContext …")

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
            train_df = train_df.copy()
            train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])

        # --- Synchronize Time ---
        # Capture the max training timestamp. We use this as "Now" during inference
        # to ensure feature distributions match the training data. 
        self.max_train_ts = train_df["last_interaction_ts"].max()

        # --- Pass min_cooccurrence as kwarg to avoid positional mismatch ---
        self.similarity_matrix = build_cooccurrence_matrix(
            interactions=train_df, 
            min_cooccurrence=min_cooccurrence
        )
        print(f"  similarity_matrix: {len(self.similarity_matrix):,} items")

        # Popularity & Stats
        pop_df = compute_item_popularity(train_df)
        
        max_score = pop_df["interaction_score"].max()
        max_count = pop_df["interaction_count"].max()
        self.item_stats_dict = {}
        for row in pop_df.itertuples():
            self.item_stats_dict[row.item_id] = {
                'score': row.interaction_score / max_score if max_score > 0 else 0,
                'count': row.interaction_count / max_count if max_count > 0 else 0
            }
            
        self.top_popular_items = pop_df["item_id"].tolist()

        # User History
        self.user_histories = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.user_last_ts = train_df.groupby("user_id")["last_interaction_ts"].max().to_dict()
        print("  RankingContext ready.")


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
def load_model(model_path: str):
    """
    Unified loader that patches scikit-learn 1.5+ attribute errors on the fly.
    """
    class _ModelWrapper:
        def __init__(self, path: str):
            if path.endswith(".joblib"):
                # 1. Load the pipeline
                self._model = joblib.load(path)

                # 2. Extract the actual LogisticRegression object
                # If it's a Pipeline, it's in named_steps['model']
                if hasattr(self._model, 'named_steps'):
                    lr_obj = self._model.named_steps['model']
                else:
                    lr_obj = self._model

                # MONKEY PATCH: If 'multi_class' is missing, add it.
                # Newer scikit-learn versions removed this, but internal 
                # functions sometimes still look for it in serialized objects. 
                if not hasattr(lr_obj, 'multi_class'):
                    lr_obj.multi_class = 'auto' 
                self._kind = "lr"
            else:
                self._model = xgb.Booster()
                self._model.load_model(path)
                self._kind = "xgb"

        def score(self, X: np.ndarray) -> np.ndarray:
            if self._kind == "lr":
                return self._model.predict_proba(X)[:, 1]
            return self._model.predict(xgb.DMatrix(X, feature_names=FEATURE_COLUMNS))

    return _ModelWrapper(model_path)


# -------------------------------------------------------
# Core Inference
# -------------------------------------------------------
def infer(user_id: int, ctx: RankingContext, model, n_candidates: int = N_CANDIDATES, k: int = K_FINAL) -> list[int]:
    """
    Full inference pipeline for a single user.
    """
    user_history = ctx.user_histories.get(user_id, set())
    user_last    = ctx.user_last_ts.get(user_id, pd.NaT)

    # --- 1. Candidate generation (top-N) ---
    candidates = recommend_item_based(
        user_history=user_history,
        similarity_matrix=ctx.similarity_matrix,
        top_popular_items=ctx.top_popular_items,
        k=n_candidates,
    )

    if not candidates:
        return []

     # --- 2. Feature computation ---
    # --- Use Frozen Time ---
    # Use the max timestamp from training, NOT the current wall-clock time.
    # This prevents 'time_since_last_interaction' from becoming massive (out of distribution).
    now = ctx.max_train_ts

    feature_rows = [
        fast_compute_features(
            row_user_id          = user_id,
            row_item_id          = item_id,
            row_ts               = now,
            row_weight           = 0.0,
            user_history_set     = user_history,
            user_last_ts         = user_last,
            similarity_matrix    = ctx.similarity_matrix,
            item_stats_dict      = ctx.item_stats_dict, 
            is_negative          = True, 
        )
        for item_id in candidates
    ]

    feature_df = pd.DataFrame(feature_rows)
    X = feature_df[FEATURE_COLUMNS].values

    ## --- 3. Score ---
    try:
        scores = model.score(X)
    except Exception as e:
        print(f"Error during scoring for user {user_id}: {e}")
        return candidates[:k] # Fallback to item-based order

    # --- 4. Sort & return top-K ---
    feature_df["_score"] = scores
    top_k = feature_df.nlargest(k, "_score")["item_id"].tolist()
    return top_k


# -------------------------------------------------------
# Batch Helper
# -------------------------------------------------------
def infer_batch(user_ids: list[int], ctx: RankingContext, model, n_candidates: int = N_CANDIDATES, k: int = K_FINAL) -> Dict[int, list[int]]:
    results = {}
    for i, uid in enumerate(user_ids):
        results[uid] = infer(uid, ctx, model, n_candidates, k)
        if (i + 1) % 1000 == 0:
            print(f"  Scored {i + 1:,}/{len(user_ids):,} users")
    return results

if __name__ == "__main__":
    from src.ranking.evaluation import time_based_split
    if os.path.exists(INTERACTIONS_PATH) and os.path.exists(XGB_MODEL_PATH):
        interactions = pd.read_csv(INTERACTIONS_PATH)
        train, _test = time_based_split(interactions)
        ctx = RankingContext(train)
        model = load_model(XGB_MODEL_PATH)
        sample_users = train["user_id"].unique()[:5].tolist()
        for uid in sample_users:
            recs = infer(uid, ctx, model)
            print(f"  user {uid} → {recs}")