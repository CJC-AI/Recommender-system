import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, recall_score, precision_score
from sklearn.pipeline import Pipeline

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
TRAINING_DATA_PATH = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH  = "artifacts/models/lr_ranker.joblib"

# LEAKAGE FIX: 'interaction_type_weight' removed from features
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "time_since_last_interaction",
]

VAL_FRACTION = 0.1  # last 10% of rows held out as validation

# -------------------------------------------------------
# Load & Split
# -------------------------------------------------------
def load_and_split(
    path: str = TRAINING_DATA_PATH,
    val_fraction: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data and split into train / val.
    Split is positional to respect temporal ordering.
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
def train_lr(train_df: pd.DataFrame) -> Pipeline:
    """
    Train a LogisticRegression ranker with scaling and sample weights.
    """
    # 1. Prepare Features (X) and Target (y)
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values

    # 2. Prepare Sample Weights (The Fix)
    # We use interaction_type_weight to value purchases higher than views.
    # CRITICAL: Fill 0.0 weights (negatives) with 1.0 so model learns from them.
    weights = train_df["interaction_type_weight"].values.copy()
    weights[weights == 0] = 1.0

    # 3. Create Pipeline (Scaling + Model)
    # StandardScaler is crucial for LR because 'time_since...' and 'popularity' 
    # have vastly different ranges.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        ))
    ])

    # 4. Fit with Sample Weights
    # Note: We pass sample_weight to the 'model' step of the pipeline
    pipeline.fit(X, y, model__sample_weight=weights)

    # Extract model to print coefficients
    model = pipeline.named_steps['model']
    print("LogisticRegression trained.")
    print(f"  Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_[0].round(4)))}")
    print(f"  Intercept:    {model.intercept_[0]:.4f}")
    
    return pipeline

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
def evaluate_lr(pipeline: Pipeline, val_df: pd.DataFrame) -> dict:
    """
    Log metrics on the validation split.
    """
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["label"].values

    # Predict Probabilities
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    # Predict Classes (Threshold 0.5)
    y_pred = pipeline.predict(X_val)

    # Calculate Metrics
    auc       = roc_auc_score(y_val, y_prob)
    logloss   = log_loss(y_val, y_prob)
    recall    = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)

    print(f"\nValidation metrics:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Log Loss:  {logloss:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")

    return {
        "auc": auc, 
        "log_loss": logloss, 
        "recall": recall, 
        "precision": precision
    }

# -------------------------------------------------------
# Save
# -------------------------------------------------------
def save_model(pipeline: Pipeline, path: str = MODEL_OUTPUT_PATH):
    """
    Persist the entire pipeline (scaler + model) via joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\nModel pipeline saved → {path}")

# -------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    train_df, val_df = load_and_split()
    model_pipeline   = train_lr(train_df)
    evaluate_lr(model_pipeline, val_df)
    save_model(model_pipeline)