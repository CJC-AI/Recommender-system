import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from typing import Dict, Set

from src.candidate_generation import (
    build_cooccurrence_matrix,
    compute_item_popularity,
    recommend_item_based,
)
from src.dataset import fast_compute_features


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
INTERACTIONS_PATH   = "data/processed/interactions.csv"
LR_MODEL_PATH       = "models/lr_ranker.joblib"
XGB_MODEL_PATH      = "models/xgb_ranker.json"

FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "time_since_last_interaction",
    "interaction_type_weight",
]

N_CANDIDATES = 50      # candidates pulled from item-item CF
K_FINAL      = 10      # final recommendations returned


# -------------------------------------------------------
# Shared State (built once, reused across calls)
# -------------------------------------------------------
class RankingContext:
    """
    Holds every pre-computed artifact needed to score candidates.
    Build once at startup; pass into infer().

    Attributes:
        similarity_matrix:    {item_id: {neighbor_id: score}}
        item_popularity_dict: {item_id: normalised popularity}
        top_popular_items:    list sorted descending by popularity
        user_histories:       {user_id: set of item_ids} from TRAIN only
        user_last_ts:         {user_id: last interaction timestamp}  from TRAIN only
    """

    def __init__(self, train_df: pd.DataFrame, min_cooccurrence: int = 2):
        print("Building RankingContext …")

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
            train_df = train_df.copy()
            train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])

        # Similarity
        self.similarity_matrix: Dict[int, Dict[int, float]] = build_cooccurrence_matrix(
            interactions=train_df,
            min_cooccurrence=min_cooccurrence,
        )
        print(f"  similarity_matrix: {len(self.similarity_matrix):,} items")

        # Popularity (normalised)
        pop_df = compute_item_popularity(train_df)
        raw    = pop_df.set_index("item_id")["interaction_score"].to_dict()
        max_pop = max(raw.values()) if raw else 1.0
        self.item_popularity_dict: Dict[int, float] = {k: v / max_pop for k, v in raw.items()}
        self.top_popular_items: list = pop_df["item_id"].tolist()   # already sorted desc

        # Per-user history & last timestamp (train only)
        self.user_histories: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_last_ts: Dict[int, pd.Timestamp] = (
            train_df.groupby("user_id")["last_interaction_ts"].max().to_dict()
        )
        print("  RankingContext ready.")


# -------------------------------------------------------
# Load a trained model (LR or XGB)
# -------------------------------------------------------
def load_model(model_path: str):
    """
    Load whichever model lives at *model_path*.

    Returns a thin wrapper that exposes a uniform .score(X) → np.ndarray
    interface so the rest of infer.py doesn't care about model type.

    Args:
        model_path: path to .joblib (LR) or .json (XGB)

    Returns:
        _ModelWrapper
    """

    class _ModelWrapper:
        """Duck-type wrapper: always exposes score(X) → 1-D array of P(positive)."""

        def __init__(self, path: str):
            if path.endswith(".joblib"):
                self._model = joblib.load(path)
                self._kind  = "lr"
            elif path.endswith(".json"):
                self._model = xgb.Booster()
                self._model.load_model(path)
                self._kind  = "xgb"
            else:
                raise ValueError(f"Unsupported model format: {path}")
            print(f"Loaded {self._kind} model from {path}")

        def score(self, X: np.ndarray) -> np.ndarray:
            if self._kind == "lr":
                return self._model.predict_proba(X)[:, 1]
            else:
                return self._model.predict(xgb.DMatrix(X))

    return _ModelWrapper(model_path)


# -------------------------------------------------------
# Core inference
# -------------------------------------------------------
def infer(
    user_id: int,
    ctx: RankingContext,
    model,                        # _ModelWrapper from load_model()
    n_candidates: int = N_CANDIDATES,
    k: int = K_FINAL,
) -> list[int]:
    """
    Full inference pipeline for a single user.

    1. Pull top-N candidates via item-item CF (popularity fallback built in).
    2. Compute the 4 ranking features for every candidate using
       fast_compute_features() from dataset.py — keeps feature logic in one place.
    3. Score with the trained model.
    4. Sort descending, return top-K item_ids.

    Args:
        user_id:      Target user
        ctx:          Pre-computed RankingContext
        model:        Loaded model wrapper
        n_candidates: How many candidates to retrieve (step 1)
        k:            How many final recommendations to return

    Returns:
        List of item_ids, length ≤ k
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
    # Use a synthetic "now" timestamp for time_since_last so the feature is
    # computed consistently with training.  If the user has no last_ts we
    # fall back to the same row timestamp (fast_compute_features handles NaT).
    now = pd.Timestamp.now()

    feature_rows = [
        fast_compute_features(
            row_user_id          = user_id,
            row_item_id          = item_id,
            row_ts               = now,
            row_weight           = 0.0,          # unknown at inference time → 0
            user_history_set     = user_history,
            user_last_ts         = user_last,
            similarity_matrix    = ctx.similarity_matrix,
            item_popularity_dict = ctx.item_popularity_dict,
            is_negative          = True,          # weight set to 0 via is_negative
        )
        for item_id in candidates
    ]

    feature_df = pd.DataFrame(feature_rows)
    X          = feature_df[FEATURE_COLUMNS].values

    # --- 3. Score ---
    scores = model.score(X)

    # --- 4. Sort & return top-K ---
    feature_df["_score"] = scores
    top_k = (
        feature_df
        .nlargest(k, "_score")["item_id"]
        .tolist()
    )
    return top_k


# -------------------------------------------------------
# Batch helper  (score every user in a DataFrame)
# -------------------------------------------------------
def infer_batch(
    user_ids: list[int],
    ctx: RankingContext,
    model,
    n_candidates: int = N_CANDIDATES,
    k: int = K_FINAL,
) -> Dict[int, list[int]]:
    """
    Run infer() for every user in the list.

    Args:
        user_ids:     List of user IDs to score
        ctx:          RankingContext
        model:        Loaded model wrapper
        n_candidates: Candidates per user
        k:            Final recs per user

    Returns:
        {user_id: [item_id, …]}  (length ≤ k per user)
    """
    results: Dict[int, list[int]] = {}
    for i, uid in enumerate(user_ids):
        results[uid] = infer(uid, ctx, model, n_candidates, k)
        if (i + 1) % 1000 == 0:
            print(f"  Scored {i + 1:,}/{len(user_ids):,} users")
    return results


# -------------------------------------------------------
# CLI Entry Point  (smoke-test with 5 users)
# -------------------------------------------------------
if __name__ == "__main__":
    from src.evaluation import time_based_split

    # 1. Load interactions → train split only
    interactions = pd.read_csv(INTERACTIONS_PATH)
    train, _test = time_based_split(interactions)

    # 2. Build shared context
    ctx = RankingContext(train)

    # 3. Load model  (swap path to LR_MODEL_PATH if preferred)
    model = load_model(XGB_MODEL_PATH)

    # 4. Run for 5 sample users
    sample_users = train["user_id"].unique()[:5].tolist()
    for uid in sample_users:
        recs = infer(uid, ctx, model)
        print(f"  user {uid} → {recs}")