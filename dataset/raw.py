import os
import pandas as pd

from utils import create_directories, timer

@timer
def process_hqa_data(input_path, output_path):
    """
    Process the HQA dataset and save selected columns to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
    """
    hqa = pd.read_csv(input_path)
    hqa[["topic", "subtopic", "question"]].to_csv(output_path, index=False)

@timer
def process_alpaca_data(input_path, output_path):
    """
    Process the Alpaca dataset and save selected columns to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
    """
    alpaca = pd.read_csv(input_path)    
    alpaca["instruction"] = alpaca["instruction"] + " " + alpaca["input"].fillna("")
    alpaca["instruction"] = alpaca["instruction"].str.strip()
    alpaca[["instruction", "output"]].to_csv(output_path, index=False)

@timer
def process_toxic_data(input_path, output_path):
    """
    Process the Toxic dataset, filter based on criteria, and save selected columns to a CSV file.

    Args:
        input_path: The path to the input CSV file.
        output_path: The path to the output CSV file.
    """
    toxic = pd.read_csv(input_path)
    toxic = toxic[(toxic["is_noise_or_spam_text"] == 0) & 
                  (toxic["related_to_election_2024"] == 0)]

    def ends_with_question(text):
        return text.endswith("?")

    toxic = toxic[toxic["text"].apply(ends_with_question)].reset_index(drop=True)
    toxic[["text", "topic", "toxicity", "threat_incitement_to_violence", 
           "insults"]].to_csv(output_path, index=False)

def raw():
    create_directories(type="raw")
    process_hqa_data("./data/landing/declare-lab_harmfulqa.csv", "data/raw/hqa.csv")
    process_alpaca_data("./data/landing/cahya_alpaca-id-cleaned.csv", "data/raw/alpaca.csv")
    process_toxic_data("./data/landing/exqrch_indotoxic2024.csv", "data/raw/toxic.csv")

if __name__ == "__main__":
    # raw()
    process_alpaca_data("./data/landing/cahya_alpaca-id-cleaned.csv", "data/raw/alpaca.csv")
