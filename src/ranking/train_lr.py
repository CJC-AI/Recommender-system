import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, recall_score, precision_score
from sklearn.pipeline import Pipeline

# --- Global Configuration ---
TRAINING_DATA_PATH = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH  = "artifacts/models/lr_ranker.joblib"

# UPDATED FEATURES
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "item_interaction_count",       # NEW
    "user_history_count",           # NEW
    "time_since_last_interaction",
]

VAL_FRACTION = 0.1 # Holds out the most recent 10% of data for validation

def load_and_split(path: str = TRAINING_DATA_PATH, val_fraction: float = VAL_FRACTION):
    """
    Loads processed training data and performs a temporal split.
    
    Args:
        path (str): Path to the CSV training data.
        val_fraction (float): Percentage of data to use for validation.
        
    Returns:
        tuple: (train_df, val_df)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run dataset.py first.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows.")

    # Split is positional rather than random to respect the time-series nature of logs
    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx].copy()
    val_df   = df.iloc[split_idx:].copy()

    return train_df, val_df

def train_lr(train_df: pd.DataFrame) -> Pipeline:
    """
    Builds and trains a scikit-learn Pipeline for ranking.
    
    Logic:
    1. Scales features using StandardScaler (required for Logistic Regression convergence).
    2. Applies sample weights (Purchases = 5x weight of a View).
    3. Fits a binary classifier to predict P(interaction).
    """
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["label"].values

    # BLOCK: Sample Weighting
    # We treat every training row with a label of 0 (negative) as a weight of 1.0.
    # Positive interactions (label 1) use their original interaction_type_weight.
    weights = train_df["interaction_type_weight"].values.copy()
    weights[weights == 0] = 1.0

    # Build the Pipeline to ensure scaling is applied during both fit and predict
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000, 
            # --- FIX C: Simplify Weights ---
            # Removed class_weight="balanced" to avoid double-weighting.
            # We rely purely on the 'weights' array (View=1 vs Purchase=5) passed to fit().
            random_state=42
        ))
    ])

    # Fit the model specifically passing weights to the 'model' step of the pipeline
    pipeline.fit(X, y, model__sample_weight=weights)
    
    # Extract model to print coefficients for verification
    model = pipeline.named_steps['model']
    print("LogisticRegression trained.")
    print(f"  Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_[0].round(4)))}")
    print(f"  Intercept:    {model.intercept_[0]:.4f}")
    
    return pipeline

def evaluate_lr(pipeline: Pipeline, val_df: pd.DataFrame):
    """Calculates ranking-relevant classification metrics on the validation set."""
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["label"].values
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    print(f"\nValidation AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"Validation Log Loss: {log_loss(y_val, y_prob):.4f}")

def save_model(pipeline: Pipeline, path: str = MODEL_OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    train, val = load_and_split()
    model = train_lr(train)
    evaluate_lr(model, val)
    save_model(model)