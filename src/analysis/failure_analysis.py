import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from typing import List, Dict

# Import existing pipeline components to ensure consistency
from src.candidate_generation import recommend_item_based
from src.ranking.dataset import fast_compute_features, load_item_metadata, build_user_category_profiles
from src.ranking.infer import RankingContext, FEATURE_COLUMNS, N_CANDIDATES
from src.ranking.evaluation import time_based_split
from src.taxonomy import TaxonomyEngine

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "interactions.csv")
LR_MODEL_PATH     = os.path.join(BASE_DIR, "artifacts", "models", "lr_ranker.joblib")
XGB_MODEL_PATH    = os.path.join(BASE_DIR, "artifacts", "models", "xgb_ranker.json")

def load_models():
    """Load both models for comparison with compatibility patches."""
    print("Loading models...")
    
    # 1. Load Logistic Regression (Pipeline)
    lr_model = joblib.load(LR_MODEL_PATH)
    
    # --- FIX: Apply Monkey Patch for Scikit-Learn Version Mismatch ---
    # The error 'LogisticRegression' object has no attribute 'multi_class' 
    # happens when loading models saved with different sklearn versions.
    if hasattr(lr_model, 'named_steps'):
        inner_estimator = lr_model.named_steps['model']
        if not hasattr(inner_estimator, 'multi_class'):
            print("  [Patching] Adding missing 'multi_class' attribute to LR model...")
            inner_estimator.multi_class = 'auto'
    
    # 2. Load XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_PATH)
    
    return lr_model, xgb_model

def get_feature_df(user_id: int, ctx: RankingContext):
    """
    Generate the full candidate feature dataframe for a specific user.
    Similar to infer.py, but returns the raw dataframe for analysis.
    """
    user_history = ctx.user_histories.get(user_id, set())
    user_last_ts = ctx.user_last_ts.get(user_id, pd.NaT)
    
    # 1. Generate Candidates
    candidates = recommend_item_based(
        user_history=user_history,
        similarity_matrix=ctx.similarity_matrix,
        top_popular_items=ctx.top_popular_items,
        k=N_CANDIDATES,
        taxonomy_engine=ctx.taxonomy_engine
    )
    
    if not candidates:
        return pd.DataFrame()
        
    # 2. Compute Features
    now = ctx.max_train_ts
    feature_rows = [
        fast_compute_features(
            row_user_id=user_id,
            row_item_id=item_id,
            row_ts=now,
            row_weight=0.0,
            user_history_set=user_history,
            user_last_ts=user_last_ts,
            similarity_matrix=ctx.similarity_matrix,
            item_stats_dict=ctx.item_stats_dict,
            item_categories=ctx.item_categories,
            user_category_profiles=ctx.user_category_profiles,
            is_negative=True
        )
        for item_id in candidates
    ]
    
    return pd.DataFrame(feature_rows)

def analyze_failure_cases(n_cases: int = 5):
    """
    Finds and explains cases where LR ranked the correct item highly, 
    but XGBoost buried it.
    """
    # 1. Setup
    if not os.path.exists(INTERACTIONS_PATH):
        print("Interactions file not found.")
        return

    interactions = pd.read_csv(INTERACTIONS_PATH)
    train, test = time_based_split(interactions)
    ctx = RankingContext(train)
    lr_model, xgb_model = load_models()
    
    # Group test data by user for ground truth
    test_history = test.groupby("user_id")["item_id"].apply(set).to_dict()
    
    print(f"\nScanning for Disagreement Cases (LR=Hit, XGB=Miss)...")
    found_cases = 0
    
    # 2. Scan Users
    # We iterate until we find N interesting failure cases
    for user_id, true_items in test_history.items():
        if found_cases >= n_cases:
            break
            
        # Generate Candidates & Features
        df = get_feature_df(user_id, ctx)
        if df.empty:
            continue
            
        # Check if true item is even in candidates
        candidates = set(df['item_id'].values)
        hits = list(true_items.intersection(candidates))
        
        if not hits:
            continue # Rankers can't help if candidate gen failed
            
        target_item = hits[0] # Analyze the first true positive found
        
        # 3. Score with Both Models
        # Ensure we pass exactly the feature columns the models expect
        X = df[FEATURE_COLUMNS]
        
        # LR Scores
        # Pipeline expects dataframe, so we pass X directly
        df['lr_score'] = lr_model.predict_proba(X)[:, 1]
        
        # XGB Scores
        # XGB expects DMatrix with specific feature names
        dtest = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
        df['xgb_score'] = xgb_model.predict(dtest)
        
        # 4. Calculate Ranks
        df['lr_rank'] = df['lr_score'].rank(ascending=False)
        df['xgb_rank'] = df['xgb_score'].rank(ascending=False)
        
        target_row = df[df['item_id'] == target_item].iloc[0]
        
        # 5. Define "Interesting Disagreement"
        # LR put it in Top 10, XGB put it outside Top 20
        if target_row['lr_rank'] <= 10 and target_row['xgb_rank'] > 20:
            found_cases += 1
            print("\n" + "="*80)
            print(f"CASE #{found_cases}: User {user_id} | Target Item {target_item}")
            print("="*80)
            print(f"  Ground Truth: User actually interacted with Item {target_item}")
            print(f"  LR Rank:      #{int(target_row['lr_rank'])}  (Score: {target_row['lr_score']:.4f}) -> SUCCESS")
            print(f"  XGB Rank:     #{int(target_row['xgb_rank'])}  (Score: {target_row['xgb_score']:.4f}) -> FAILURE")
            
            print("\n  --- WHY THEY DISAGREED (Feature Analysis) ---")
            print(f"  Item Similarity:       {target_row['item_similarity_score']:.4f}  (High similarity usually drives LR)")
            print(f"  Interaction Count:     {target_row['item_interaction_count']:.4f}  (Low count usually scares XGB)")
            print(f"  Category Affinity:     {target_row['user_category_affinity']:.4f}")
            print(f"  Item Popularity:       {target_row['item_popularity']:.4f}")
            
            print("\n  --- WHAT XGBOOST PREFERRED INSTEAD ---")
            # Show top 3 items XGBoost liked more
            xgb_top = df.sort_values('xgb_rank').head(3)
            # Use column selection that guarantees existence
            cols_to_show = ['item_id', 'item_interaction_count', 'item_similarity_score', 'user_category_affinity', 'xgb_score']
            print(xgb_top[cols_to_show])
            
            print("\n  [Insight]: Notice if XGBoost's top picks have much higher 'Interaction Count' or 'Affinity'?")

if __name__ == "__main__":
    if os.path.exists(LR_MODEL_PATH) and os.path.exists(XGB_MODEL_PATH):
        analyze_failure_cases()
    else:
        print("Please train both models before running analysis.")