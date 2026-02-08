import pandas as pd
import numpy as np
import xgboost as xgb
import math


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
TRAINING_DATA_PATH  = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH   = "artifacts/models/xgb_ranker.json"

FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "time_since_last_interaction",
    "interaction_type_weight",
]

VAL_FRACTION = 0.1

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             42,
    "verbosity":        0,
}

NUM_BOOST_ROUNDS = 200
EARLY_STOP_ROUNDS = 20


# -------------------------------------------------------
# Load & Split
# -------------------------------------------------------
def load_and_split(
    path: str = TRAINING_DATA_PATH,
    val_fraction: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data and split into train / val.

    Positional split preserves temporal ordering from dataset.py.

    Args:
        path: Path to training_data.csv
        val_fraction: Fraction held out for validation

    Returns:
        (train_df, val_df)
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Positive ratio: {df['label'].mean():.3f}")

    split_idx = int(len(df) * (1 - val_fraction))
    train_df  = df.iloc[:split_idx].copy()
    val_df    = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    return train_df, val_df


# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train_xgb(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> xgb.Booster:
    """
    Train an XGBoost binary:logistic ranker with early stopping on val.

    Args:
        train_df: Training split
        val_df:   Validation split (used for early stopping only)

    Returns:
        Trained xgb.Booster
    """
    dtrain = xgb.DMatrix(
        train_df[FEATURE_COLUMNS].values,
        label=train_df["label"].values,
    )
    dval = xgb.DMatrix(
        val_df[FEATURE_COLUMNS].values,
        label=val_df["label"].values,
    )

    evals_log: list = []

    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        evals_log=evals_log,
        verbose_eval=50,
    )

    print(f"\nXGBoost trained.  Best iteration: {model.best_iteration}")
    print(f"  Train log-loss: {evals_log['train']['logloss'][model.best_iteration]:.4f}")
    print(f"  Val   log-loss: {evals_log['val']['logloss'][model.best_iteration]:.4f}")
    return model


# -------------------------------------------------------
# NDCG@K helper
# -------------------------------------------------------
def _dcg(relevances: list) -> float:
    """Compute DCG for a ranked list of binary relevances."""
    return sum(
        rel / math.log2(i + 2)          # i is 0-indexed → denominator = log2(rank+1)
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@K for a single user's candidate list.

    Args:
        y_true:  Binary ground-truth labels (1 = relevant)
        y_score: Predicted scores (higher = more relevant)
        k:       Cut-off rank

    Returns:
        NDCG@K score (0.0–1.0)
    """
    # Sort by predicted score descending, take top-k
    order   = np.argsort(-y_score)[:k]
    ranked  = y_true[order].tolist()

    dcg  = _dcg(ranked)

    # Ideal: all 1 s first
    ideal = sorted(y_true.tolist(), reverse=True)[:k]
    idcg  = _dcg(ideal)

    return dcg / idcg if idcg > 0 else 0.0


# -------------------------------------------------------
# Evaluate NDCG@10
# -------------------------------------------------------
def evaluate_xgb(model: xgb.Booster, val_df: pd.DataFrame, k: int = 10) -> dict:
    """
    Score every user in val, group by user_id, compute per-user NDCG@K,
    then return the macro-average.

    Args:
        model: Trained Booster
        val_df: Validation DataFrame (must contain user_id, label, FEATURE_COLUMNS)
        k:      Rank cut-off

    Returns:
        Dict with 'ndcg_at_10' (macro-averaged over users)
    """
    dval = xgb.DMatrix(val_df[FEATURE_COLUMNS].values)
    val_df = val_df.copy()
    val_df["score"] = model.predict(dval)

    ndcgs = []
    for user_id, group in val_df.groupby("user_id"):
        if group["label"].sum() == 0:
            # No positive example for this user in val → skip
            continue
        ndcgs.append(
            ndcg_at_k(
                y_true  = group["label"].values,
                y_score = group["score"].values,
                k=k,
            )
        )

    mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    print(f"Validation NDCG@{k}: {mean_ndcg:.4f}  (over {len(ndcgs)} users)")
    return {"ndcg_at_10": mean_ndcg}


# -------------------------------------------------------
# Save
# -------------------------------------------------------
def save_model(model: xgb.Booster, path: str = MODEL_OUTPUT_PATH):
    """
    Persist XGBoost model as JSON.

    Args:
        model: Trained Booster
        path:  Output path
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"Model saved → {path}")


# -------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    train_df, val_df = load_and_split()
    model            = train_xgb(train_df, val_df)
    evaluate_xgb(model, val_df)
    save_model(model)