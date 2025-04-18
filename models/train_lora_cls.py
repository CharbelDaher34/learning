"""
Train a LoRA adapter for classification on the AG News dataset.
"""

import os
import torch
import pandas as pd
from pathlib import Path
import argparse
import mlflow
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftType,
)
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
from datasets import Dataset
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .base_model import get_model_and_tokenizer, get_model_info, DEFAULT_MODEL_NAME
from .mlflow_utils import (
    setup_mlflow,
    get_or_create_experiment,
    log_model_params,
    log_model_metrics,
    register_model,
    transition_model_to_production,
    MLFLOW_EXPERIMENT_CLS,
)

# Default training parameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_MAX_SEQ_LENGTH = 128
NUM_CLASSES = 4  # AG News has 4 classes


class ClassificationModel(torch.nn.Module):
    """Classification model with a LoRA-tuned backbone and a classification head."""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_classes = num_classes

        # Classification head
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the last hidden state from base model
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs
        )

        # Get the last hidden state of the [CLS] token (first token)
        last_hidden_state = outputs.hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]

        # Pass through the classifier
        logits = self.classifier(cls_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> Dataset:
    """
    Prepare the dataset for classification.

    Args:
        data_path: Path to the parquet file
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length

    Returns:
        Dataset object
    """
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # For classification, we'll use both title and description
    df["text"] = df["title"] + " " + df["description"]

    # Convert labels to zero-indexed (1-4 to 0-3)
    df["label_idx"] = df["label"] - 1

    # Convert to Dataset
    dataset = Dataset.from_pandas(df[["text", "label_idx"]])

    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
    )

    # Rename label_idx to labels for the trainer
    tokenized_dataset = tokenized_dataset.rename_column("label_idx", "labels")

    # Remove original text column
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    return tokenized_dataset


def compute_metrics(eval_pred):
    """
    Compute classification metrics.

    Args:
        eval_pred: Evaluation predictions

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_lora_adapter(
    model_name: str = DEFAULT_MODEL_NAME,
    train_data_path: str = "etl/data/ag_news_train.parquet",
    val_data_path: str = "etl/data/ag_news_val.parquet",
    output_dir: str = "models/lora-cls",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """
    Train a LoRA adapter for classification.

    Args:
        model_name: Name of the Hugging Face model
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Directory to save the adapter
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_seq_length: Maximum sequence length
        num_classes: Number of classes for classification

    Returns:
        Dictionary with training results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    setup_mlflow()
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT_CLS)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # Log parameters
        params = {
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "num_classes": num_classes,
        }
        log_model_params(params)

        # Load model and tokenizer
        base_model, tokenizer = get_model_and_tokenizer(model_name)

        # Prepare model for LoRA fine-tuning
        base_model = prepare_model_for_kbit_training(base_model)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # Extract features for classification
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        )

        # Apply LoRA to the base model
        base_model = get_peft_model(base_model, lora_config)

        # Wrap with classification head
        model = ClassificationModel(base_model, num_classes)

        # Prepare datasets
        train_dataset = prepare_dataset(train_data_path, tokenizer, max_seq_length)
        val_dataset = prepare_dataset(val_data_path, tokenizer, max_seq_length)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            report_to="none",  # Disable default Wandb/tensorboard reporting
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()

        # Log metrics
        log_model_metrics(eval_results)

        # Save the full model with classification head
        full_model_path = os.path.join(output_dir, "full_model")
        trainer.save_model(full_model_path)

        # Save the adapter separately
        adapter_path = os.path.join(output_dir, "adapter")
        base_model.save_pretrained(adapter_path)

        # Also save the classifier weights
        torch.save(model.classifier.state_dict(), os.path.join(adapter_path, "classifier.pt"))

        # Log the adapter to MLflow
        mlflow.log_artifact(adapter_path, "adapter")
        mlflow.log_artifact(full_model_path, "full_model")

        # Register the model in MLflow Model Registry
        registered_model_name = "lora-cls"
        model_uri = f"runs:/{run_id}/adapter"
        version = register_model(
            model_uri=model_uri,
            name=registered_model_name,
            description=f"LoRA adapter for classification fine-tuned on AG News dataset using {model_name}",
            tags={
                "task": "classification",
                "base_model": model_name,
                "dataset": "ag_news",
                "num_classes": str(num_classes),
            },
        )

        # Consider transitioning to production
        transition_model_to_production(registered_model_name, version)

        # Return results
        return {
            "run_id": run_id,
            "model_version": version,
            "metrics": eval_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for classification")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Base model name"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="etl/data/ag_news_train.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="etl/data/ag_news_val.parquet",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/lora-cls", help="Output directory"
    )
    parser.add_argument(
        "--lora_r", type=int, default=DEFAULT_LORA_R, help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum sequence length"
    )

    args = parser.parse_args()

    result = train_lora_adapter(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
    )

    print(f"Training complete. Run ID: {result['run_id']}")
    print(f"Model version: {result['model_version']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
