"""
Utilities for MLflow tracking and model versioning.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Tuple
import datetime

# Default MLflow settings
DEFAULT_TRACKING_URI = "mlflow"
MLFLOW_EXPERIMENT_GEN = "lora-gen"
MLFLOW_EXPERIMENT_CLS = "lora-cls"


def setup_mlflow(tracking_uri: str = DEFAULT_TRACKING_URI) -> None:
    """
    Set up MLflow tracking.

    Args:
        tracking_uri: URI for MLflow tracking server
    """
    mlflow.set_tracking_uri(tracking_uri)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an MLflow experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def log_model_params(params: Dict[str, Any]) -> None:
    """
    Log model parameters to MLflow.

    Args:
        params: Dictionary of model parameters
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_model_metrics(metrics: Dict[str, float]) -> None:
    """
    Log model metrics to MLflow.

    Args:
        metrics: Dictionary of metrics
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log an artifact to MLflow.

    Args:
        local_path: Local path to the artifact
        artifact_path: Path within the artifact store
    """
    mlflow.log_artifact(local_path, artifact_path)


def register_model(
    model_uri: str,
    name: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Register a model in the MLflow Model Registry.

    Args:
        model_uri: URI to the model
        name: Name to register the model under
        description: Optional description
        tags: Optional tags

    Returns:
        Model version
    """
    client = MlflowClient()

    # Register the model
    result = mlflow.register_model(model_uri, name)
    version = result.version

    # Set description if provided
    if description:
        client.update_model_version(name=name, version=version, description=description)

    # Set tags if provided
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name=name, version=version, key=key, value=value)

    return version


def transition_model_to_production(name: str, version: str, archive_existing: bool = True) -> None:
    """
    Transition a model version to production.

    Args:
        name: Model name
        version: Model version
        archive_existing: Whether to archive existing production models
    """
    client = MlflowClient()

    # Archive existing production models if requested
    if archive_existing:
        for model_version in client.search_model_versions(f"name='{name}'"):
            if model_version.current_stage == "Production":
                client.transition_model_version_stage(
                    name=name, version=model_version.version, stage="Archived"
                )

    # Transition new model to production
    client.transition_model_version_stage(name=name, version=version, stage="Production")


def get_latest_model_version(name: str, stage: str = "Production") -> Optional[str]:
    """
    Get the latest model version in the specified stage.

    Args:
        name: Model name
        stage: Model stage (e.g., "Production", "Staging")

    Returns:
        Model version or None if no model found
    """
    client = MlflowClient()

    for model_version in client.search_model_versions(f"name='{name}'"):
        if model_version.current_stage == stage:
            return model_version.version

    return None


def get_model_metrics(name: str, version: str, metric_keys: List[str]) -> Dict[str, float]:
    """
    Get metrics for a specific model version.

    Args:
        name: Model name
        version: Model version
        metric_keys: List of metric keys to retrieve

    Returns:
        Dictionary of metrics
    """
    client = MlflowClient()

    # Get run ID for the model version
    model_version = client.get_model_version(name, version)
    run_id = model_version.run_id

    # Get metrics from the run
    metrics = {}
    for key in metric_keys:
        try:
            metric = client.get_metric_history(run_id, key)
            if metric:
                metrics[key] = metric[-1].value
        except:
            # Metric not found
            pass

    return metrics


def compare_model_versions(
    name: str,
    current_version: str,
    new_metrics: Dict[str, float],
    primary_metric: str,
    higher_is_better: bool = True,
) -> Tuple[bool, float, float]:
    """
    Compare a new model's metrics with the current production model.

    Args:
        name: Model name
        current_version: Current model version
        new_metrics: New model's metrics
        primary_metric: Primary metric to compare
        higher_is_better: Whether higher values are better for the primary metric

    Returns:
        Tuple of (is_better, current_metric, new_metric)
    """
    # Get current model metrics
    current_metrics = get_model_metrics(name, current_version, [primary_metric])

    # If current model has no metrics, assume new model is better
    if primary_metric not in current_metrics:
        return True, 0.0, new_metrics.get(primary_metric, 0.0)

    current_metric = current_metrics[primary_metric]
    new_metric = new_metrics.get(primary_metric, 0.0)

    # Compare metrics
    if higher_is_better:
        is_better = new_metric > current_metric
    else:
        is_better = new_metric < current_metric

    return is_better, current_metric, new_metric
