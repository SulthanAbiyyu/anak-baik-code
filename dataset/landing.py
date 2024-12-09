import os
import pandas as pd
from datasets import load_dataset
from utils import create_directories, timer

# List of repositories and their dataset splits
hf_repos = [
    ("declare-lab/HarmfulQA", "train"),
    ("Exqrch/IndoToxic2024", "main"),
    ("cahya/alpaca-id-cleaned", "train")
]

@timer
def save_dataset_to_csv(dataset, split_name, filename):
    """
    Save a dataset split to a CSV file.

    Args:
        dataset: The dataset object.
        split_name: The name of the dataset split (e.g., 'train').
        filename: The path to the output CSV file.
    """
    df = pd.DataFrame(dataset[split_name])
    df.to_csv(filename, index=False)
    print(f"Saved {split_name} split to {filename}")

@timer
def load_and_save_datasets():
    """
    Load datasets from specified repositories and save them to CSV files.
    """
    for repo, split in hf_repos:
        if "Toxic" in repo:
            dataset = load_dataset(repo, split)
        else:
            dataset = load_dataset(repo)
        filename = f"data/landing/{repo.replace('/', '_').lower()}.csv"
        save_dataset_to_csv(dataset, split, filename)

def landing():
    create_directories(type="landing")
    load_and_save_datasets()

if __name__ == "__main__":
    landing()