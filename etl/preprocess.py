"""
Preprocess the AG News dataset by cleaning text and splitting into train/val/test sets.
"""

import os
import re
import pandas as pd
from pathlib import Path
import string
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

# Define AG News class labels
AG_NEWS_CLASSES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, special characters, etc.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and digits
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(input_file: str, output_dir: str, val_size: float = 0.1) -> Dict[str, str]:
    """
    Preprocess dataset and split into train/val/test sets.

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed files
        val_size: Proportion of training data to use for validation

    Returns:
        Dictionary with paths to output files
    """
    print(f"Preprocessing data from {input_file}...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read data
    df = pd.read_csv(input_file)

    # Clean text (combine title and description)
    df["text"] = df["title"] + " " + df["description"]
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Create class name column
    df["class_name"] = df["label"].map(AG_NEWS_CLASSES)

    # Split training data into train and validation
    if "train" in input_file:
        train_df, val_df = train_test_split(
            df, test_size=val_size, random_state=42, stratify=df["label"]
        )

        # Save validation set
        val_path = os.path.join(output_dir, "ag_news_val.parquet")
        val_df.to_parquet(val_path, index=False)
        print(f"Saved {len(val_df)} validation samples to {val_path}")

        # Save training set
        train_path = os.path.join(output_dir, "ag_news_train.parquet")
        train_df.to_parquet(train_path, index=False)
        print(f"Saved {len(train_df)} training samples to {train_path}")

        output_paths = {"train": train_path, "val": val_path}
    else:
        # For test set, just save the processed data
        test_path = os.path.join(output_dir, "ag_news_test.parquet")
        df.to_parquet(test_path, index=False)
        print(f"Saved {len(df)} test samples to {test_path}")

        output_paths = {"test": test_path}

    # Save class mapping
    class_map_path = os.path.join(output_dir, "class_mapping.csv")
    pd.DataFrame(list(AG_NEWS_CLASSES.items()), columns=["label", "class_name"]).to_csv(
        class_map_path, index=False
    )

    return output_paths


def process_ag_news(data_dir: str = "etl/data") -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Process both train and test datasets.

    Args:
        data_dir: Directory containing the raw and processed data

    Returns:
        Tuple of dictionaries with paths to output files
    """
    train_input = os.path.join(data_dir, "ag_news_train.csv")
    test_input = os.path.join(data_dir, "ag_news_test.csv")

    # Process training data (creates train and val splits)
    train_outputs = preprocess_dataset(train_input, data_dir)

    # Process test data
    test_outputs = preprocess_dataset(test_input, data_dir)

    # Combine outputs
    all_outputs = {**train_outputs, **test_outputs}

    print("Preprocessing complete.")
    return all_outputs


if __name__ == "__main__":
    process_ag_news()
