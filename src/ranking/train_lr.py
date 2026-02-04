import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
TRAINING_DATA_PATH = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH  = "models/lr_ranker.joblib"

FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "time_since_last_interaction",
    "interaction_type_weight",
]

VAL_FRACTION = 0.1          # last 10 % of rows held out as validation


# -------------------------------------------------------
# Load & Split
# -------------------------------------------------------
def load_and_split(
    path: str = TRAINING_DATA_PATH,
    val_fraction: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data and split into train / val.

    Split is positional (last val_fraction rows → val) so that
    the temporal ordering baked in by dataset.py is respected.

    Args:
        path: Path to training_data.csv
        val_fraction: Fraction of rows to hold out for validation

    Returns:
        (train_df, val_df)
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"  Positive ratio: {df['label'].mean():.3f}")

    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx].copy()
    val_df   = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    return train_df, val_df


# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train_lr(train_df: pd.DataFrame) -> LogisticRegression:
    """
    Train a LogisticRegression ranker on the training split.

    Args:
        train_df: DataFrame that contains FEATURE_COLUMNS and 'label'

    Returns:
        Fitted LogisticRegression model
    """
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)

    print("LogisticRegression trained.")
    print(f"  Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_[0].round(4)))}")
    print(f"  Intercept:    {model.intercept_[0]:.4f}")
    return model


# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
def evaluate_lr(model: LogisticRegression, val_df: pd.DataFrame) -> dict:
    """
    Log AUC-ROC and log-loss on the validation split.

    Args:
        model: Fitted LogisticRegression
        val_df: Validation DataFrame

    Returns:
        Dict with 'auc' and 'log_loss'
    """
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["label"].values

    y_prob = model.predict_proba(X_val)[:, 1]

    auc      = roc_auc_score(y_val, y_prob)
    logloss  = log_loss(y_val, y_prob)

    print(f"Validation metrics:")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Log-loss: {logloss:.4f}")
    return {"auc": auc, "log_loss": logloss}


# -------------------------------------------------------
# Save
# -------------------------------------------------------
def save_model(model: LogisticRegression, path: str = MODEL_OUTPUT_PATH):
    """
    Persist model to disk via joblib.

    Args:
        model: Fitted LogisticRegression
        path: Output file path
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


# -------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    train_df, val_df = load_and_split()
    model            = train_lr(train_df)
    evaluate_lr(model, val_df)
    save_model(model)