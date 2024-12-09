import os
import pandas as pd

from utils import create_directories, timer

@timer
def split_data(df, test_size=0.2):
    """
    Shuffle and split the dataset into train and test sets.

    Args:
        df: The DataFrame to split.
        test_size: The proportion of the dataset to include in the test split.

    Returns:
        A DataFrame with an added 'split' column indicating 'train' or 'test'.
    """
    df = df.sample(frac=1).reset_index(drop=True)
    split_index = int(df.shape[0] * test_size)
    df["split"] = "train"
    df.loc[:split_index, "split"] = "test"
    return df

@timer
def unify():
    """
    Process datasets, split into train and test sets, and save the final curated dataset.
    """
    alpaca = pd.read_csv("data/transformed/alpaca.csv")
    toxic = pd.read_csv("data/transformed/toxic.csv")
    hqa = pd.read_csv("data/transformed/hqa_translated.csv")

    # Split the datasets
    alpaca = split_data(alpaca)
    toxic = split_data(toxic)
    hqa = split_data(hqa)

    # Combine datasets and shuffle
    final = pd.concat([alpaca, toxic, hqa])
    final = final.sample(frac=1).reset_index(drop=True)
    final.to_csv("data/curated/final.csv", index=False)

def curate():
    create_directories(type="curated")
    unify()

if __name__ == "__main__":
    curate()
