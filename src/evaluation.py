import pandas as pd

def time_based_split(
        interactions: pd.DataFrame,
        timestamp_col: str = 'last_interaction_ts',
        train_ratio: float = 0.8,
):
    """
    Split interactions into train and test sets by time.
    
    Train: first train_ratio%
    Test: remaining interactions
    """

    # Sort interactions by timestamp
    interactions = interactions.sort_values(timestamp_col).reset_index(drop=True)

    # Find split index
    split_index = int(len(interactions) * train_ratio)

    # Split the data
    train = interactions.iloc[:split_index].copy()
    test = interactions.iloc[split_index:].copy()

    return train, test