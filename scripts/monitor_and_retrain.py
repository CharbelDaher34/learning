"""
Monitor model performance and trigger retraining when needed.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import mlflow
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from models import (
    load_classification_model_from_mlflow,
    load_generation_model_from_mlflow,
    get_latest_model_version,
    get_model_metrics,
    compare_model_versions,
    MLFLOW_EXPERIMENT_GEN,
    MLFLOW_EXPERIMENT_CLS,
)


def evaluate_classification_model(
    model_path: Optional[str] = None,
    data_path: str = "etl/data/ag_news_val.parquet",
    model_name: str = "lora-cls",
    stage: str = "Production",
) -> Dict[str, float]:
    """
    Evaluate the classification model on the validation dataset.

    Args:
        model_path: Path to the model, if None will load from MLflow
        data_path: Path to the validation data
        model_name: Name of the registered model
        stage: Model stage to evaluate

    Returns:
        Dictionary of metrics
    """
    print(f"Evaluating classification model on {data_path}")

    # Load the data
    df = pd.read_parquet(data_path)

    # Prepare texts and labels
    texts = df["title"] + " " + df["description"]
    labels = df["label"]

    # Sample a subset for faster evaluation if dataset is large
    if len(texts) > 1000:
        sampled_indices = df.sample(1000, random_state=42).index
        texts = texts.loc[sampled_indices]
        labels = labels.loc[sampled_indices]

    # Load the model
    if model_path:
        # TODO: Implement loading from local path if needed
        raise NotImplementedError("Loading from local path not implemented yet")
    else:
        model = load_classification_model_from_mlflow(model_name, stage)

    # Evaluate the model
    predictions = model.predict(texts.tolist())

    # Calculate metrics
    correct = 0
    for pred, true_label in zip(predictions, labels):
        if pred["label"] == true_label:
            correct += 1

    accuracy = correct / len(labels)

    # Return metrics
    return {"accuracy": accuracy}


def evaluate_generation_model(
    model_path: Optional[str] = None,
    data_path: str = "etl/data/ag_news_val.parquet",
    model_name: str = "lora-gen",
    stage: str = "Production",
    sample_size: int = 10,
) -> Dict[str, float]:
    """
    Evaluate the generation model on the validation dataset.

    Args:
        model_path: Path to the model, if None will load from MLflow
        data_path: Path to the validation data
        model_name: Name of the registered model
        stage: Model stage to evaluate
        sample_size: Number of samples to evaluate

    Returns:
        Dictionary of metrics
    """
    # For generation, we'll use perplexity as the main metric
    # This is calculated during training and stored in MLflow
    # Here we're just retrieving it

    if model_path:
        # TODO: Implement loading metrics from local path
        raise NotImplementedError("Loading from local path not implemented yet")
    else:
        # Get the latest version
        version = get_latest_model_version(model_name, stage)
        if not version:
            return {"perplexity": float("inf")}

        # Get the metrics
        metrics = get_model_metrics(model_name, version, ["perplexity"])

        return metrics


def should_retrain(
    model_type: str,
    new_metrics: Dict[str, float],
    threshold_improvement: float = 0.01,
) -> Tuple[bool, str]:
    """
    Determine if retraining is needed based on metrics.

    Args:
        model_type: Type of model ("classification" or "generation")
        new_metrics: New evaluation metrics
        threshold_improvement: Minimum improvement threshold

    Returns:
        Tuple of (should_retrain, reason)
    """
    if model_type == "classification":
        model_name = "lora-cls"
        primary_metric = "accuracy"
        higher_is_better = True
    else:
        model_name = "lora-gen"
        primary_metric = "perplexity"
        higher_is_better = False

    # Get the latest version
    version = get_latest_model_version(model_name, "Production")
    if not version:
        return True, f"No existing {model_type} model found"

    # Compare with current model
    is_better, current_value, new_value = compare_model_versions(
        model_name, version, new_metrics, primary_metric, higher_is_better
    )

    # Calculate improvement
    if higher_is_better:
        improvement = new_value - current_value
        relative_improvement = improvement / current_value if current_value > 0 else float("inf")
    else:
        improvement = current_value - new_value
        relative_improvement = improvement / current_value if current_value > 0 else float("inf")

    # Check if improvement exceeds threshold
    if is_better and relative_improvement >= threshold_improvement:
        return (
            True,
            f"{primary_metric} improved from {current_value:.4f} to {new_value:.4f} ({relative_improvement:.2%})",
        )

    return (
        False,
        f"Improvement not significant: {primary_metric} changed from {current_value:.4f} to {new_value:.4f}",
    )


def trigger_retraining(
    model_type: str,
    data_dir: str = "etl/data",
    output_dir: Optional[str] = None,
) -> int:
    """
    Trigger retraining of a model.

    Args:
        model_type: Type of model ("classification" or "generation")
        data_dir: Directory containing the data
        output_dir: Output directory for the model

    Returns:
        Return code from the subprocess
    """
    if model_type == "classification":
        script = "models.train_lora_cls"
        default_output_dir = "models/lora-cls"
    else:
        script = "models.train_lora_gen"
        default_output_dir = "models/lora-gen"

    output_dir = output_dir or default_output_dir

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare command
    train_data = os.path.join(data_dir, "ag_news_train.parquet")
    val_data = os.path.join(data_dir, "ag_news_val.parquet")

    command = [
        sys.executable,
        "-m",
        script,
        "--train_data",
        train_data,
        "--val_data",
        val_data,
        "--output_dir",
        output_dir,
    ]

    # Execute the command
    print(f"Triggering retraining of {model_type} model")
    print(f"Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    # Print output
    print(f"Return code: {result.returncode}")
    print(f"Output:\n{result.stdout}")

    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")

    return result.returncode


def monitor_and_retrain(
    model_type: str,
    data_dir: str = "etl/data",
    force_retrain: bool = False,
    threshold_improvement: float = 0.01,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Monitor model performance and retrain if needed.

    Args:
        model_type: Type of model ("classification" or "generation")
        data_dir: Directory containing the data
        force_retrain: Whether to force retraining regardless of metrics
        threshold_improvement: Minimum improvement threshold
        output_dir: Output directory for the model

    Returns:
        Dictionary with monitoring results
    """
    # Validate model type
    if model_type not in ["classification", "generation"]:
        raise ValueError(f"Invalid model type: {model_type}")

    # Evaluate current model
    val_data = os.path.join(data_dir, "ag_news_val.parquet")

    if model_type == "classification":
        metrics = evaluate_classification_model(data_path=val_data)
    else:
        metrics = evaluate_generation_model(data_path=val_data)

    print(f"Current {model_type} model metrics: {metrics}")

    # Check if retraining is needed
    needs_retrain, reason = should_retrain(model_type, metrics, threshold_improvement)

    # If forcing retraining, override the decision
    if force_retrain:
        needs_retrain = True
        reason = "Retraining forced by user"

    print(f"Retraining needed: {needs_retrain} - {reason}")

    # Trigger retraining if needed
    if needs_retrain:
        retrain_result = trigger_retraining(model_type, data_dir, output_dir)

        return {
            "model_type": model_type,
            "metrics": metrics,
            "retraining_triggered": True,
            "retraining_reason": reason,
            "retraining_success": retrain_result == 0,
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "model_type": model_type,
        "metrics": metrics,
        "retraining_triggered": False,
        "retraining_reason": reason,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor model performance and trigger retraining")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["classification", "generation"],
        required=True,
        help="Type of model to monitor",
    )
    parser.add_argument(
        "--data_dir", type=str, default="etl/data", help="Directory containing the data"
    )
    parser.add_argument(
        "--force_retrain", action="store_true", help="Force retraining regardless of metrics"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Minimum improvement threshold for retraining"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for the model"
    )

    args = parser.parse_args()

    result = monitor_and_retrain(
        model_type=args.model_type,
        data_dir=args.data_dir,
        force_retrain=args.force_retrain,
        threshold_improvement=args.threshold,
        output_dir=args.output_dir,
    )

    # Print results
    print("\nMonitoring Results:")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
