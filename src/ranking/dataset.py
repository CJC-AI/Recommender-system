import pandas as pd
import numpy as np
from typing import Dict, Set, List
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
    item_stats_dict: Dict[int, dict], # Updated to accept dict of stats
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
    
    # 3. Retrieve Item Stats (Pop + Count)
    # item_stats_dict is now {item_id: {'score': 0.5, 'count': 120}}
    stats = item_stats_dict.get(row_item_id, {'score': 0.0, 'count': 0.0})

    return {
        "user_id": row_user_id,
        "item_id": row_item_id,
        "item_similarity_score": max_sim,
        "item_popularity": stats['score'],       # Normalized Score
        "item_interaction_count": stats['count'], # NEW: Raw Volume
        "user_history_count": len(user_history_set), # NEW: Warm vs Cold signal
        "time_since_last_interaction": max(0.0, time_diff),
        "interaction_type_weight": 0.0 if is_negative else get_interaction_type_weight(row_weight),
        "label": 0 if is_negative else 1
    }

# -------------------------------------------------------
# Build Training Data
# -------------------------------------------------------
def build_training_data(
    train_df: pd.DataFrame,
    sample_users_frac: float = 0.2, 
    sample_items_top_n: int = 5000,    
    n_negatives: int = 6,              
    min_cooccurrence: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build training data with advanced negative sampling.
    
    Negative Strategy:
    1. Hard Negatives: Items similar to user's history (Candidate Gen output).
    2. Popular Negatives: High-traffic items.
    3. Exclusions: Items user interacted with in Past OR Future.
    """
    
    print(f"Building TRAINING data with Enhanced Features...")
    rng = np.random.RandomState(seed)
    
    # --- 1. Pre-processing ---
    train_df = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(train_df["last_interaction_ts"]):
        print("  Converting timestamps...")
        train_df["last_interaction_ts"] = pd.to_datetime(train_df["last_interaction_ts"])

    # User Sampling
    unique_users = train_df["user_id"].unique()
    sampled_users = rng.choice(
        unique_users, 
        size=int(len(unique_users) * sample_users_frac), 
        replace=False
    )
    train_df = train_df[train_df["user_id"].isin(set(sampled_users))].sort_values("last_interaction_ts")
    print(f"  Sampled {len(sampled_users)} users.")

    # --- 2. Build Artifacts ---
    print("  Building Matrix and Global History...")
    
    # Similarity Matrix (for Hard Negatives)
    similarity_matrix = build_cooccurrence_matrix(
        interactions=train_df, 
        min_cooccurrence=min_cooccurrence
    )
    
    # Popularity (for Popular Negatives - Proxy for 'up to time t')
    pop_df = compute_item_popularity(train_df)
    
    # Normalization (Max-Scale)
    max_score = pop_df["interaction_score"].max()
    max_count = pop_df["interaction_count"].max()
    
    # Create a fast lookup dict: {item_id: {'score': norm_score, 'count': norm_count}}
    item_stats_dict = {}
    for row in pop_df.itertuples():
        item_stats_dict[row.item_id] = {
            'score': row.interaction_score / max_score if max_score > 0 else 0,
            'count': row.interaction_count / max_count if max_count > 0 else 0
        }
    
    # Sample from the FULL item list to avoid penalizing popular items exclusively.
    # This ensures the model sees popular items as both Positives (labels=1) and Negatives (labels=0).
    all_items_universe = pop_df["item_id"].values 

    # GLOBAL History (Past + Future) for strict exclusions
    # We must not sample an item as negative if the user will interact with it later
    user_global_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # --- 3. Row Generation ---
    print(f"  Generating rows (Target: {n_negatives} negatives per positive)...")
    training_data = []
    
    # Running state
    user_running_history = {} 
    user_last_ts = {}         
    
    # Using itertuples for speed
    for row in train_df.itertuples(index=False):
        u_id = row.user_id
        i_id = row.item_id
        score = row.interaction_score
        ts = row.last_interaction_ts
        
        # Get State
        current_history = user_running_history.get(u_id, set())
        last_ts = user_last_ts.get(u_id, pd.NaT)
        full_history_exclusion = user_global_history.get(u_id, set())
        
        # --- A. Positive Sample ---
        pos_row = fast_compute_features(
            u_id, i_id, ts, score, 
            current_history, last_ts, 
            similarity_matrix, item_stats_dict, 
            is_negative=False
        )
        training_data.append(pos_row)
        
        # --- B. Negative Samples ---
        selected_negatives = []
        
        # Source 1: Hard Negatives (from Candidate Gen / Neighbors)
        # We look at items similar to what the user has recently seen
        hard_candidates = set()
        if current_history:
            # Sample up to 3 recent items to find neighbors for
            # (Converting set to list is O(N), but history is usually small)
            seed_items = rng.choice(list(current_history), size=min(len(current_history), 3), replace=False)
            
            for seed in seed_items:
                neighbors = similarity_matrix.get(seed, {})
                # Add top neighbors as hard negative candidates
                if neighbors:
                    # Sort by similarity and take top 5 per seed
                    sorted_neighbors = sorted(neighbors, key=neighbors.get, reverse=True)[:5]
                    hard_candidates.update(sorted_neighbors)
        
        # Try to fill half the quota with Hard Negatives
        hard_quota = n_negatives // 2
        hard_candidates_list = list(hard_candidates)
        rng.shuffle(hard_candidates_list)
        
        for cand in hard_candidates_list:
            if len(selected_negatives) >= hard_quota: break
            # Strict Exclusion: Not in Past, Current, or Future
            if cand not in full_history_exclusion and cand != i_id:
                selected_negatives.append(cand)
                
        # Source 2: General/Popular Negatives (Backfill the rest)
        # We sample more than needed to account for collisions
        needed = n_negatives - len(selected_negatives)
        if needed > 0:
            # --- Use the full universe ---
            pop_candidates = rng.choice(all_items_universe, size=needed * 3)
            for cand in pop_candidates:
                if len(selected_negatives) >= n_negatives: break
                # Deduplicate and Exclude
                if cand not in full_history_exclusion and cand != i_id and cand not in selected_negatives:
                    selected_negatives.append(cand)
        
        # --- C. Negative Features ---
        for neg_id in selected_negatives:
            neg_row = fast_compute_features(
                u_id, neg_id, ts, 0.0, 
                current_history, last_ts, 
                similarity_matrix, item_stats_dict, 
                is_negative=True
            )
            training_data.append(neg_row)
            
        # --- D. Update State ---
        if u_id not in user_running_history:
            user_running_history[u_id] = set()
        user_running_history[u_id].add(i_id)
        user_last_ts[u_id] = ts

    df_final = pd.DataFrame(training_data)
    print(f"  Final Dataset: {len(df_final)} rows.")
    return df_final


def save_training_data(training_df: pd.DataFrame, output_path: str):
    training_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    INTERACTIONS_PATH = "data/processed/interactions.csv"
    TRAINING_OUTPUT_PATH = "data/processed/training_data.csv"
    
    if pd.io.common.file_exists(INTERACTIONS_PATH):
        interactions = pd.read_csv(INTERACTIONS_PATH)
        train, test = time_based_split(interactions)
        training_df = build_training_data(train, n_negatives=6)
        save_training_data(training_df, TRAINING_OUTPUT_PATH)
    else:
        print("Interactions file not found.")