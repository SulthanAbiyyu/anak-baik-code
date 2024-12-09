import os
import pandas as pd

from prompts import random_prompt_choices
from utils import create_directories, timer

@timer
def process_hqa_data(input_path, output_path):
    """
    Process the HQA dataset and save transformed data to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
    """
    hqa = pd.read_csv(input_path)
    hqa["topic"] = hqa["topic"].apply(lambda x: x.lower())
    hqa["instruction"] = hqa["question"]

    hqa = hqa.drop(columns=["subtopic", "question"])
    hqa["output"] = ""
    hqa["output"] = hqa["output"].apply(lambda x: random_prompt_choices("harmful"))

    hqa_topics_dict = {
        'literature and language': 'sastra dan bahasa',
        'education and pedagogy': 'pendidikan',
        'social sciences': 'ilmu sosial',
        'mathematics and logic': 'matematika',
        'health and medicine': 'kesehatan',
        'philosophy and ethics': 'filosofi',
        'science and technology': 'sains dan teknologi',
        'business and economics': 'bisnis dan ekonomi',
        'geography and environmental studies': 'geografi',
        'history and culture': 'sejarah',
    }

    hqa["topic"] = hqa["topic"].apply(lambda x: hqa_topics_dict.get(x, "Unknown"))
    hqa["type"] = "umum"

    hqa.to_csv(output_path, index=False)

@timer
def process_toxic_data(input_path, output_path):
    """
    Process the Toxic dataset and save transformed data to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
    """
    toxic = pd.read_csv(input_path)
    toxic["instruction"] = toxic["text"]
    toxic["output"] = ""
    toxic["output"] = toxic["output"].apply(lambda x: random_prompt_choices("toxic"))

    toxic = toxic.drop(columns=["text"])
    toxic["insults"] = toxic["insults"]
    toxic["insults"] = toxic["insults"].apply(lambda x: x - 1 if x > 1 else x)
    toxic["type"] = "harmful"

    def toxic_type(row):
        if row["toxicity"] == 1:
            return "kasar"
        elif row["threat_incitement_to_violence"] == 1:
            return "ancaman"
        elif row["insults"] == 1:
            return "hinaan"
        else:
            return "umum"

    toxic["type"] = toxic.apply(toxic_type, axis=1)
    toxic = toxic.drop(columns=["toxicity", "threat_incitement_to_violence", "insults"])

    toxic.to_csv(output_path, index=False)

@timer
def process_alpaca_data(input_path, output_path, total_harmful_rows):
    """
    Process the Alpaca dataset and save transformed data to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
        total_harmful_rows: The number of rows to sample from the Alpaca dataset.
    """
    alpaca = pd.read_csv(input_path)
    alpaca = alpaca.sample(n=total_harmful_rows, replace=True)
    alpaca["type"] = "harmless"
    alpaca["topic"] = "None"
    alpaca.to_csv(output_path, index=False)

def transform():
    create_directories(type="transformed")
    
    process_hqa_data("data/raw/hqa.csv", "data/transformed/hqa.csv")
    process_toxic_data("data/raw/toxic.csv", "data/transformed/toxic.csv")

    total_harmful_rows = pd.read_csv("data/raw/toxic.csv").shape[0] + pd.read_csv("data/raw/hqa.csv").shape[0]
    process_alpaca_data("data/raw/alpaca.csv", "data/transformed/alpaca.csv", total_harmful_rows)

if __name__ == "__main__":
    # transform()
    total_harmful_rows = pd.read_csv("data/raw/toxic.csv").shape[0] + pd.read_csv("data/raw/hqa.csv").shape[0]
    process_alpaca_data("data/raw/alpaca.csv", "data/transformed/alpaca.csv", total_harmful_rows)