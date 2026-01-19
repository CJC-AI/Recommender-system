import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}


# -----------------------------
# Core Functions
# -----------------------------
def load_events(path: str) -> pd.DataFrame:
    """
    Load RetailRocket events dataset.
    """
    df = pd.read_csv(path)
    return df


def map_event_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map event types to interaction weights.
    """
    df = df.copy()
    df["interaction_weight"] = df["event"].map(EVENT_WEIGHTS)
    return df


def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate interactions per (user_id, item_id).
    """
    interactions = (
        df.groupby(["visitorid", "itemid"])
        .agg(
            interaction_score=("interaction_weight", "sum"),
            last_interaction_ts=("timestamp", "max"),
        )
        .reset_index()
    )

    interactions.rename(columns={"visitorid": "user_id","itemid": "item_id",},
                        inplace=True,)

    return interactions


def build_interactions(events_path: str, output_path: str):
    """
    Full pipeline: load → map → aggregate → save
    """
    df = load_events(events_path)
    df = map_event_weights(df)
    interactions = aggregate_interactions(df)

    interactions.to_csv(output_path, index=False)
    print(f"Saved interactions to {output_path}") 


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    EVENTS_PATH = "data/raw/events.csv"
    OUTPUT_PATH = "data/processed/interactions.csv"

    build_interactions(EVENTS_PATH, OUTPUT_PATH)
