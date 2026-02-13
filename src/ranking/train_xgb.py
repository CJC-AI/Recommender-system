import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, log_loss, 
                             recall_score, precision_score, 
                             classification_report)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
TRAINING_DATA_PATH = "data/processed/training_data.csv"
MODEL_OUTPUT_PATH  = "artifacts/models/xgb_ranker.json"


# NO LEAKAGE: interaction_type_weight is excluded from features
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "item_interaction_count",
    "user_history_count",
    "time_since_last_interaction",
    
]

# XGBoost Hyperparameters (tuned for Ranking/Imbalance)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,              # Deep enough to capture non-linear interaction
    "subsample": 0.8,            # Prevent overfitting
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1                 # Use all CPU cores
}

# -------------------------------------------------------
# Load & Split
# -------------------------------------------------------
def load_and_prep(path: str = TRAINING_DATA_PATH):
    """
    Load data and prepare features, targets, and weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
        
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows.")
    print(f"  Positive ratio: {df['label'].mean():.3f}")

    # 1. Features & Target
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 2. Sample Weights (Crucial Step)
    # Map negatives (weight=0.0) to 1.0 so the model learns from them.
    # Keep positives at their original weight (View=1, Purchase=5)
    weights = df["interaction_type_weight"].values.copy()
    weights[weights == 0] = 1.0

    # 3. Stratified Split
    # We use stratified split here to ensure Validation has same Pos/Neg ratio
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.1, stratify=y, random_state=42
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    return X_train, X_val, y_train, y_val, w_train, w_val

# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train_xgboost(X_train, y_train, w_train, X_val, y_val):
    """
    Train XGBoost with Early Stopping and Sample Weights.
    """
    print("\nTraining XGBoost Ranker...")
    
    model = xgb.XGBClassifier(
        **XGB_PARAMS,
        early_stopping_rounds=50,
        )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    print("Training Complete.")
    return model

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
def evaluate_model(model, X_val, y_val):
    """
    Detailed evaluation including Feature Importance.
    """
    # Predict Probabilities (for Ranking/AUC)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Predict Classes (Threshold 0.5 for precision/recall metrics)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    logloss = log_loss(y_val, y_prob)
    
    print("\n" + "="*40)
    print("FINAL VALIDATION METRICS")
    print("="*40)
    print(f"  AUC:       {auc:.4f}")
    print(f"  Log Loss:  {logloss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Plot Feature Importance
    print("\nGenerating Feature Importance Plot...")
    
    xgb.plot_importance(model, importance_type='gain', max_num_features=10, title='Feature Importance (Gain)')
    plt.tight_layout()
    
    return auc

# -------------------------------------------------------
# Main Execution
# -------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    X_train, X_val, y_train, y_val, w_train, w_val = load_and_prep()
    
    # 2. Train
    model = train_xgboost(X_train, y_train, w_train, X_val, y_val)
    
    # 3. Evaluate
    evaluate_model(model, X_val, y_val)
    
    # 4. Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save_model(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")