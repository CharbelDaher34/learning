"""
Streamlit dashboard for monitoring, retraining, and testing models.
"""

import os
import sys
import json
import time
import requests
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import MLflow utilities
from mlflow.tracking import MlflowClient
import mlflow

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl.preprocess import AG_NEWS_CLASSES
from models.mlflow_utils import (
    setup_mlflow,
    get_latest_model_version,
    MLFLOW_EXPERIMENT_GEN,
    MLFLOW_EXPERIMENT_CLS,
)

# Constants
API_HOST = os.environ.get("API_HOST", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlflow")

# Page configuration
st.set_page_config(
    page_title="AG News LLM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def setup():
    """Set up MLflow and other configurations."""
    setup_mlflow(MLFLOW_TRACKING_URI)


def get_model_versions() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all model versions from MLflow.

    Returns:
        Dictionary mapping model name to list of versions
    """
    client = MlflowClient()
    models = {}

    # Get classification models
    try:
        cls_versions = client.search_model_versions("name='lora-cls'")
        models["classification"] = [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "creation_timestamp": datetime.fromtimestamp(
                    v.creation_timestamp / 1000.0
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for v in cls_versions
        ]
    except:
        models["classification"] = []

    # Get generation models
    try:
        gen_versions = client.search_model_versions("name='lora-gen'")
        models["generation"] = [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "creation_timestamp": datetime.fromtimestamp(
                    v.creation_timestamp / 1000.0
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for v in gen_versions
        ]
    except:
        models["generation"] = []

    return models


def get_run_metrics(run_id: str) -> Dict[str, float]:
    """
    Get metrics for a specific run.

    Args:
        run_id: The MLflow run ID

    Returns:
        Dictionary of metrics
    """
    client = MlflowClient()
    metrics = {}

    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
    except:
        pass

    return metrics


def get_experiment_metrics(experiment_id: str, metric_keys: List[str]) -> pd.DataFrame:
    """
    Get metrics for all runs in an experiment.

    Args:
        experiment_id: The MLflow experiment ID
        metric_keys: List of metric keys to retrieve

    Returns:
        DataFrame with metrics for each run
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attribute.start_time DESC"],
    )

    metrics_data = []
    for run in runs:
        run_metrics = {
            "run_id": run.info.run_id,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0),
        }

        # Get parameters
        for key, value in run.data.params.items():
            run_metrics[f"param_{key}"] = value

        # Get metrics
        for key in metric_keys:
            if key in run.data.metrics:
                run_metrics[key] = run.data.metrics[key]

        metrics_data.append(run_metrics)

    return pd.DataFrame(metrics_data)


def trigger_monitoring(model_type: str, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Trigger model monitoring and potential retraining.

    Args:
        model_type: Type of model ("classification" or "generation")
        force_retrain: Whether to force retraining

    Returns:
        Dictionary with monitoring results
    """
    command = [
        sys.executable,
        "-m",
        "scripts.monitor_and_retrain",
        "--model_type",
        model_type,
    ]

    if force_retrain:
        command.append("--force_retrain")

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Parse the output
    output = result.stdout

    # Extract the monitoring results
    monitoring_results = {}
    try:
        lines = output.split("\n")
        results_section = False
        results_text = ""

        for line in lines:
            if "Monitoring Results:" in line:
                results_section = True
                continue

            if results_section:
                results_text += line + "\n"

        # Try to parse the results as JSON
        if results_text:
            # Extract key-value pairs
            results_dict = {}
            for line in results_text.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    results_dict[key.strip()] = value.strip()

            monitoring_results = results_dict
    except Exception as e:
        monitoring_results = {"error": str(e), "output": output}

    return monitoring_results


def call_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the API endpoint.

    Args:
        endpoint: API endpoint
        data: Request data

    Returns:
        API response
    """
    url = f"{API_HOST}/{endpoint}"

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def monitoring_tab():
    """Render the monitoring tab."""
    st.header("📊 Model Monitoring")

    # Get model versions
    model_versions = get_model_versions()

    # Create tabs for different models
    model_tabs = st.tabs(["Classification Model", "Generation Model"])

    # Classification model tab
    with model_tabs[0]:
        st.subheader("Classification Model Metrics")

        # Model versions table
        if model_versions["classification"]:
            st.write("### Model Versions")
            st.dataframe(pd.DataFrame(model_versions["classification"]))

            # Get metrics for each version
            metrics_data = []
            for version in model_versions["classification"]:
                metrics = get_run_metrics(version["run_id"])
                metrics["version"] = version["version"]
                metrics["stage"] = version["stage"]
                metrics["creation_timestamp"] = version["creation_timestamp"]
                metrics_data.append(metrics)

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)

                # Plot accuracy over versions
                if "accuracy" in metrics_df.columns:
                    st.write("### Accuracy by Version")
                    fig = px.line(
                        metrics_df,
                        x="version",
                        y="accuracy",
                        markers=True,
                        title="Model Accuracy by Version",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Plot all metrics
                st.write("### All Metrics")
                st.dataframe(metrics_df)
        else:
            st.info("No classification models found in MLflow. Train a model first.")

        # Get experiment metrics
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_CLS)
        if experiment:
            metrics_df = get_experiment_metrics(
                experiment.experiment_id, ["accuracy", "f1", "precision", "recall"]
            )

            if not metrics_df.empty:
                st.write("### Training Runs")
                st.dataframe(
                    metrics_df[["run_id", "start_time", "accuracy", "f1", "precision", "recall"]]
                )

                # Plot metrics over time
                st.write("### Metrics Over Time")
                fig = px.line(
                    metrics_df,
                    x="start_time",
                    y=["accuracy", "f1", "precision", "recall"],
                    title="Classification Metrics Over Time",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Generation model tab
    with model_tabs[1]:
        st.subheader("Generation Model Metrics")

        # Model versions table
        if model_versions["generation"]:
            st.write("### Model Versions")
            st.dataframe(pd.DataFrame(model_versions["generation"]))

            # Get metrics for each version
            metrics_data = []
            for version in model_versions["generation"]:
                metrics = get_run_metrics(version["run_id"])
                metrics["version"] = version["version"]
                metrics["stage"] = version["stage"]
                metrics["creation_timestamp"] = version["creation_timestamp"]
                metrics_data.append(metrics)

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)

                # Plot perplexity over versions
                if "perplexity" in metrics_df.columns:
                    st.write("### Perplexity by Version")
                    fig = px.line(
                        metrics_df,
                        x="version",
                        y="perplexity",
                        markers=True,
                        title="Model Perplexity by Version (lower is better)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Plot all metrics
                st.write("### All Metrics")
                st.dataframe(metrics_df)
        else:
            st.info("No generation models found in MLflow. Train a model first.")

        # Get experiment metrics
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_GEN)
        if experiment:
            metrics_df = get_experiment_metrics(experiment.experiment_id, ["loss", "perplexity"])

            if not metrics_df.empty:
                st.write("### Training Runs")
                st.dataframe(metrics_df[["run_id", "start_time", "loss", "perplexity"]])

                # Plot metrics over time
                st.write("### Metrics Over Time")
                fig = px.line(
                    metrics_df,
                    x="start_time",
                    y=["perplexity"],
                    title="Generation Perplexity Over Time (lower is better)",
                )
                st.plotly_chart(fig, use_container_width=True)


def retrain_tab():
    """Render the retraining tab."""
    st.header("🔄 Continual Learning & Retraining")

    st.markdown(
        """
    This tab allows you to:
    - Monitor the latest model performance
    - Manually trigger retraining when needed
    - View model comparison with previous versions
    """
    )

    # Model selection
    model_type = st.radio(
        "Select Model Type",
        ["classification", "generation"],
        format_func=lambda x: (
            "Classification Model" if x == "classification" else "Generation Model"
        ),
    )

    # Force retraining option
    force_retrain = st.checkbox("Force Retraining", value=False)

    # Trigger monitoring and retraining
    if st.button("Evaluate and Retrain (if needed)"):
        with st.spinner(f"Evaluating {model_type} model..."):
            results = trigger_monitoring(model_type, force_retrain)

            # Display results
            st.json(results)

            # Show success/failure message
            if "retraining_triggered" in results and results["retraining_triggered"] == "True":
                if "retraining_success" in results and results["retraining_success"] == "True":
                    st.success("Retraining completed successfully!")
                else:
                    st.error("Retraining failed. Check logs for details.")
            else:
                st.info("No retraining needed based on current metrics.")


def tester_tab():
    """Render the model tester tab."""
    st.header("🧪 Model Tester")

    st.markdown(
        """
    This tab allows you to test the deployed models with your own inputs.
    """
    )

    # Create tabs for different models
    model_tabs = st.tabs(["Classification", "Text Generation"])

    # Classification tester
    with model_tabs[0]:
        st.subheader("News Classification")

        # Text input
        text_input = st.text_area(
            "Enter text to classify:",
            height=150,
            placeholder="Enter news text to classify...",
        )

        # Test button
        if st.button("Classify"):
            if text_input:
                with st.spinner("Classifying..."):
                    # Call the API
                    response = call_api("classify", {"text": text_input})

                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        # Display results
                        st.write(f"**Model Version:** {response.get('model_version', 'Unknown')}")
                        st.write(
                            f"**Processing Time:** {response.get('processing_time', 0):.3f} seconds"
                        )

                        results = response.get("results", [])
                        if results:
                            result = results[0]

                            # Show primary classification
                            st.markdown(
                                f"### Primary Classification: **{result.get('class_name', 'Unknown')}**"
                            )

                            # Show probabilities
                            probs = result.get("probabilities", {})
                            probs_df = pd.DataFrame(
                                {
                                    "Category": list(probs.keys()),
                                    "Probability": list(probs.values()),
                                }
                            )
                            probs_df = probs_df.sort_values("Probability", ascending=False)

                            # Plot probabilities
                            fig = px.bar(
                                probs_df,
                                x="Category",
                                y="Probability",
                                title="Classification Probabilities",
                                color="Probability",
                                color_continuous_scale="blues",
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")

    # Generation tester
    with model_tabs[1]:
        st.subheader("News Text Generation")

        # Prompt input
        prompt = st.text_area(
            "Enter a prompt:",
            height=100,
            placeholder="Write a news article about...",
        )

        # Category selection
        category = st.selectbox(
            "News Category (Optional)",
            [None] + list(AG_NEWS_CLASSES.values()),
        )

        # Generation parameters
        col1, col2 = st.columns(2)
        with col1:
            max_tokens = st.slider("Max New Tokens", 10, 500, 100)
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
        with col2:
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
            top_k = st.slider("Top-k", 1, 100, 50)

        # Test button
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating..."):
                    # Prepare request
                    request = {
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }

                    if category:
                        request["category"] = category

                    # Call the API
                    response = call_api("generate", request)

                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        # Display results
                        st.write(f"**Model Version:** {response.get('model_version', 'Unknown')}")
                        st.write(
                            f"**Processing Time:** {response.get('processing_time', 0):.3f} seconds"
                        )

                        results = response.get("results", {})
                        if results:
                            if isinstance(results, list):
                                results = results[0]  # Take the first result

                            # Show generated text
                            st.markdown("### Generated Text:")
                            st.markdown(f"#### Prompt:")
                            st.markdown(f"> {prompt}")
                            st.markdown(f"#### Generated Content:")
                            st.markdown(f"{results.get('generated_text', '')}")
            else:
                st.warning("Please enter a prompt for generation.")


def main():
    """Main function for the Streamlit app."""
    # Set up MLflow
    setup()

    # Sidebar
    st.sidebar.title("AG News LLM Dashboard")
    st.sidebar.image(
        "https://www.shutterstock.com/shutterstock/photos/2308261631/display_1500/stock-vector-digital-news-logo-design-vector-template-2308261631.jpg",
        width=200,
    )

    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["Monitor", "Retrain", "Test"],
        format_func=lambda x: (
            f"📊 {x}" if x == "Monitor" else (f"🔄 {x}" if x == "Retrain" else f"🧪 {x}")
        ),
    )

    # Render appropriate tab based on selection
    if page == "Monitor":
        monitoring_tab()
    elif page == "Retrain":
        retrain_tab()
    elif page == "Test":
        tester_tab()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("AG News LLM Project - v0.1.0")


def start():
    """Entry point for running the Streamlit app."""
    import streamlit.web.cli as stcli
    import sys

    sys.argv = ["streamlit", "run", __file__, "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
