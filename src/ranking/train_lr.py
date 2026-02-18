import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, log_loss, 
                             classification_report)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "training_data.csv")
MODEL_OUTPUT_PATH  = os.path.join(BASE_DIR, "artifacts", "models", "lr_ranker.joblib")

# THE FULL 6-FEATURE SET
FEATURE_COLUMNS = [
    "item_similarity_score",
    "item_popularity",
    "item_interaction_count",       
    "user_history_count",           
    "user_category_affinity",       # New Personalization Feature
    "time_since_last_interaction",  
]

# -------------------------------------------------------
# Load & Split
# -------------------------------------------------------
def load_and_prep(path: str = TRAINING_DATA_PATH):
    """
    Load data, handle weights, and split.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")
        
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows.")
    
    # Check for missing columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}. Re-run dataset.py!")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Sample Weights: Negatives (0.0) -> 1.0, Positives -> Original Weight (1, 3, 5)
    weights = df["interaction_type_weight"].values.copy()
    weights[weights == 0] = 1.0

    # Stratified Split
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.1, stratify=y, random_state=42
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    return X_train, X_val, y_train, y_val, w_train, w_val

# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train_lr(X_train, y_train, w_train):
    """
    Train Logistic Regression with Scaling.
    """
    print("\nTraining Logistic Regression Ranker...")
    
    # Pipeline: Scale features -> LogReg
    # Scaling is MANDATORY because 'item_interaction_count' (0-1000s) 
    # has a vastly different scale than 'item_popularity' (0-1).
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42
        ))
    ])
    
    pipe.fit(X_train, y_train, model__sample_weight=w_train)
    
    print("Training Complete.")
    return pipe

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
def evaluate_model(model, X_val, y_val):
    """
    Evaluate AUC and print coefficients (Feature Importance).
    """
    y_prob = model.predict_proba(X_val)[:, 1]
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

    # Print Coefficients (Linear Feature Importance)
    print("\nFeature Coefficients (Importance):")
    print("-" * 30)
    
    # Extract coefficients from the inner model
    coeffs = model.named_steps['model'].coef_[0]
    feature_imp = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Coefficient': coeffs,
        'Abs_Coeff': np.abs(coeffs)
    }).sort_values(by='Abs_Coeff', ascending=False)
    
    print(feature_imp[['Feature', 'Coefficient']])

# -------------------------------------------------------
# Main Execution
# -------------------------------------------------------
if __name__ == "__main__":
    # 1. Load
    X_train, X_val, y_train, y_val, w_train, w_val = load_and_prep()
    
    # 2. Train
    model = train_lr(X_train, y_train, w_train)
    
    # 3. Evaluate
    evaluate_model(model, X_val, y_val)
    
    # 4. Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")