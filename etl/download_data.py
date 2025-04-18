"""
Download AG News dataset from Hugging Face and save it to csv files.
"""

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def download_ag_news(output_dir: str = "etl/data") -> None:
    """
    Downloads the AG News dataset and saves it to CSV files.

    Args:
        output_dir: Directory to save the dataset files
    """
    print("Downloading AG News dataset...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset("ag_news")

    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Save to CSV files
    train_path = os.path.join(output_dir, "ag_news_train.csv")
    test_path = os.path.join(output_dir, "ag_news_test.csv")

    print(f"Saving training data to {train_path}")
    train_df.to_csv(train_path, index=False)

    print(f"Saving test data to {test_path}")
    test_df.to_csv(test_path, index=False)

    print(f"Dataset downloaded and saved successfully.")
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")


if __name__ == "__main__":
    download_ag_news()
