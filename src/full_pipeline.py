"""
full_pipeline.py
================
End-to-end recommendation pipeline.  Run the whole thing from scratch:

    python full_pipeline.py

Or skip stages whose artefacts already exist on disk:

    python full_pipeline.py --skip interactions --skip train_data

Stages (in order):
    1. interactions   – raw events  →  interactions.csv
    2. train_data     – interactions.csv  →  training_data.csv
    3. train_lr       – training_data.csv →  models/lr_ranker.joblib
    4. train_xgb      – training_data.csv →  models/xgb_ranker.json
    5. evaluate       – test-set comparison of all three systems
"""

import argparse
import os
import pandas as pd


# -------------------------------------------------------
# Configuration  (single source of truth for every path)
# -------------------------------------------------------
EVENTS_PATH          = "data/raw/events.csv"
INTERACTIONS_PATH    = "data/processed/interactions.csv"
TRAINING_DATA_PATH   = "data/processed/training_data.csv"
LR_MODEL_PATH        = "models/lr_ranker.joblib"
XGB_MODEL_PATH       = "models/xgb_ranker.json"

ALL_STAGES = ["interactions", "train_data", "train_lr", "train_xgb", "evaluate"]


# -------------------------------------------------------
# Stage 1: Build interactions
# -------------------------------------------------------
def stage_interactions(skip: bool = False):
    """
    Raw events CSV  ->  aggregated interactions CSV.
    Uses interactions.py from the existing pipeline.
    """
    if skip:
        print("[SKIP] interactions – already exists")
        return

    print("\n" + "=" * 60)
    print(" STAGE 1 / 5 –  BUILD INTERACTIONS")
    print("=" * 60)

    from src.interactions import build_interactions
    build_interactions(EVENTS_PATH, INTERACTIONS_PATH)


# -------------------------------------------------------
# Stage 2: Build training data
# -------------------------------------------------------
def stage_train_data(skip: bool = False):
    """
    interactions.csv  ->  training_data.csv  (sampled positive/negative pairs).
    Uses dataset.py from the existing pipeline.
    """
    if skip:
        print("[SKIP] train_data – already exists")
        return

    print("\n" + "=" * 60)
    print(" STAGE 2 / 5 –  BUILD TRAINING DATA")
    print("=" * 60)

    from src.dataset    import build_training_data, save_training_data
    from src.evaluation import time_based_split

    interactions = pd.read_csv(INTERACTIONS_PATH)
    train, _test = time_based_split(interactions)

    training_df = build_training_data(train)
    save_training_data(training_df, TRAINING_DATA_PATH)


# -------------------------------------------------------
# Stage 3: Train Logistic Regression
# -------------------------------------------------------
def stage_train_lr(skip: bool = False):
    """
    training_data.csv  ->  models/lr_ranker.joblib
    """
    if skip:
        print("[SKIP] train_lr – already exists")
        return

    print("\n" + "=" * 60)
    print(" STAGE 3 / 5 –  TRAIN LOGISTIC REGRESSION")
    print("=" * 60)

    from src.train_lr import load_and_split, train_lr, evaluate_lr, save_model

    train_df, val_df = load_and_split(TRAINING_DATA_PATH)
    model            = train_lr(train_df)
    evaluate_lr(model, val_df)
    save_model(model, LR_MODEL_PATH)


# -------------------------------------------------------
# Stage 4: Train XGBoost
# -------------------------------------------------------
def stage_train_xgb(skip: bool = False):
    """
    training_data.csv  ->  models/xgb_ranker.json
    """
    if skip:
        print("[SKIP] train_xgb – already exists")
        return

    print("\n" + "=" * 60)
    print(" STAGE 4 / 5 –  TRAIN XGBOOST")
    print("=" * 60)

    from src.train_xgb import load_and_split, train_xgb, evaluate_xgb, save_model

    train_df, val_df = load_and_split(TRAINING_DATA_PATH)
    model            = train_xgb(train_df, val_df)
    evaluate_xgb(model, val_df)
    save_model(model, XGB_MODEL_PATH)


# -------------------------------------------------------
# Stage 5: Evaluate  (all three systems, test set)
# -------------------------------------------------------
def stage_evaluate(model_path: str = XGB_MODEL_PATH):
    """
    Run compare_models() which evaluates Popularity, Item-Item CF,
    and Item-Item CF + Ranker on the held-out test split.
    """
    print("\n" + "=" * 60)
    print(" STAGE 5 / 5 –  EVALUATION")
    print("=" * 60)

    from src.evaluation import compare_models

    interactions = pd.read_csv(INTERACTIONS_PATH)
    compare_models(interactions, model_path=model_path)


# -------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end recommendation pipeline."
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=ALL_STAGES,
        help="Stage name to skip (repeatable).  Example: --skip interactions --skip train_data",
    )
    parser.add_argument(
        "--model",
        default=XGB_MODEL_PATH,
        help="Which trained model to use for evaluation (default: xgb_ranker.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    skip = set(args.skip)

    print("=" * 60)
    print(" FULL PIPELINE")
    print("=" * 60)
    print(f"  Stages to run : {[s for s in ALL_STAGES if s not in skip]}")
    if skip:
        print(f"  Stages skipped: {list(skip)}")

    stage_interactions(skip="interactions" in skip)
    stage_train_data(  skip="train_data"   in skip)
    stage_train_lr(    skip="train_lr"     in skip)
    stage_train_xgb(   skip="train_xgb"   in skip)

    if "evaluate" not in skip:
        stage_evaluate(model_path=args.model)
    else:
        print("[SKIP] evaluate")

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)