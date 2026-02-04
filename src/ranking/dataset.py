import pandas as pd
import numpy as np
from typing import Dict, Set
from src.candidate_generation import build_cooccurrence_matrix, compute_item_popularity
from src.ranking.evaluation import time_based_split


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
INTERACTION_WEIGHTS = {
    1.0: 1.0,   # view
    3.0: 3.0,   # addtocart
    5.0: 5.0,   # transaction
}

def get_interaction_type_weight(interaction_score: float) -> float:
    return INTERACTION_WEIGHTS.get(interaction_score, interaction_score)

# -------------------------------------------------------
# Fast Feature Calculation
# -------------------------------------------------------
def fast_compute_features(
    row_user_id: int,
    row_item_id: int,
    row_ts: pd.Timestamp,
    row_weight: float,
    user_history_set: Set[int],
    user_last_ts: pd.Timestamp,
    similarity_matrix: Dict[int, Dict[int, float]],
    item_popularity_dict: Dict[int, float],
    is_negative: bool = False
) -> dict:
    
    # 1. Similarity Score (Optimized Intersection)
    max_sim = 0.0
    if user_history_set:
        neighbors = similarity_matrix.get(row_item_id, {})
        if neighbors:
            # Fast set intersection
            common_items = user_history_set.intersection(neighbors.keys())
            if common_items:
                max_sim = max(neighbors[i] for i in common_items)

    # 2. Time Since Last Interaction
    time_diff = 0.0
    if not pd.isnull(user_last_ts):
        time_diff = (row_ts - user_last_ts).total_seconds() / 3600.0
        
    return {
        "user_id": row_user_id,
        "item_id": row_item_id,
        "item_similarity_score": max_sim,
        "item_popularity": item_popularity_dict.get(row_item_id, 0.0),
        "time_since_last_interaction": max(0.0, time_diff),
        "interaction_type_weight": 0.0 if is_negative else get_interaction_type_weight(row_weight),
        "label": 0 if is_negative else 1
    }

# -------------------------------------------------------
# Build Training Data
# -------------------------------------------------------
def build_training_data(
    train_df: pd.DataFrame,
    sample_users_frac: float = 0.2,    # Sample 20% of users
    sample_items_top_n: int = 5000,    # Negative samples from top 5k items
    n_negatives: int = 2,              # 2 negatives per positive
    min_cooccurrence: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    
    print(f"Building SAMPLING-OPTIMIZED training data...")
    rng = np.random.RandomState(seed)
    
    # Ensure timestamps
    train_df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
        print("  Converting timestamps to datetime...")
        train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])

    # 1. User Sampling
    unique_users = train_df["user_id"].unique()
    sampled_users = rng.choice(
        unique_users, 
        size=int(len(unique_users) * sample_users_frac), 
        replace=False
    )
    train_df = train_df[train_df["user_id"].isin(set(sampled_users))].sort_values("last_interaction_ts")
    print(f"  Sampled {len(sampled_users)} users for training.")

    # 2. Build Artifacts
    print("  Building similarity matrix & popularity dict...")
    
    # --- FIX: Use Keyword Argument for min_cooccurrence ---
    similarity_matrix = build_cooccurrence_matrix(
        interactions=train_df, 
        min_cooccurrence=min_cooccurrence
    )
    # -----------------------------------------------------
    
    pop_df = compute_item_popularity(train_df)
    item_popularity_dict = pop_df.set_index("item_id")["interaction_score"].to_dict()
    
    # Normalize Popularity
    max_pop = max(item_popularity_dict.values()) if item_popularity_dict else 1.0
    item_popularity_dict = {k: v / max_pop for k, v in item_popularity_dict.items()}
    
    # Universe for Negatives
    top_items_universe = pop_df.head(sample_items_top_n)["item_id"].values
    
    # 3. Generate Rows
    print(f"  Generating rows...")
    training_data = []
    user_running_history = {} 
    user_last_ts = {}         
    
    for row in train_df.itertuples(index=False):
        # Assumes columns: user_id, item_id, interaction_score, last_interaction_ts
        u_id = row.user_id
        i_id = row.item_id
        score = row.interaction_score
        ts = row.last_interaction_ts
        
        history = user_running_history.get(u_id, set())
        last_ts = user_last_ts.get(u_id, pd.NaT)
        
        # Positive
        pos_row = fast_compute_features(
            u_id, i_id, ts, score, 
            history, last_ts, 
            similarity_matrix, item_popularity_dict, 
            is_negative=False
        )
        training_data.append(pos_row)
        
        # Negatives (Vectorized Sampling)
        candidates = rng.choice(top_items_universe, size=n_negatives * 2) 
        neg_count = 0
        for neg_id in candidates:
            if neg_count >= n_negatives: break
            if neg_id == i_id or neg_id in history: continue
                
            neg_row = fast_compute_features(
                u_id, neg_id, ts, 0.0, 
                history, last_ts, 
                similarity_matrix, item_popularity_dict, 
                is_negative=True
            )
            training_data.append(neg_row)
            neg_count += 1
            
        # Update State
        if u_id not in user_running_history:
            user_running_history[u_id] = set()
        user_running_history[u_id].add(i_id)
        user_last_ts[u_id] = ts

    df_final = pd.DataFrame(training_data)
    print(f"  Final Dataset: {len(df_final)} rows.")
    return df_final


def save_training_data(training_df: pd.DataFrame, output_path: str):
    """Save training data to CSV."""
    training_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")



# -------------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    INTERACTIONS_PATH = "data/processed/interactions.csv"
    TRAINING_OUTPUT_PATH = "data/processed/training_data.csv"

    # Load interactions
    interactions = pd.read_csv(INTERACTIONS_PATH)

    # Split into train/test
    train, test = time_based_split(interactions)

    # Build training data
    training_df = build_training_data(train)

    # Save training data
    save_training_data(training_df, TRAINING_OUTPUT_PATH)