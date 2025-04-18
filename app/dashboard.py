import streamlit as st
import pandas as pd
import requests
import mlflow
from datetime import datetime, timedelta
import plotly.express as px
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import os

# Configure page
st.set_page_config(
    page_title="AG News ML Pipeline Dashboard", page_icon="📊", layout="wide"
)

# Constants
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "your-api-key-here")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_mlflow_metrics(days=7):
    """Get metrics from MLflow for the last N days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    metrics = {
        "generation": {"loss": [], "timestamp": []},
        "classification": {"f1": [], "accuracy": [], "timestamp": []},
    }

    for task in ["generation", "classification"]:
        runs = mlflow.search_runs(
            experiment_ids=[
                mlflow.get_experiment_by_name(f"lora_{task}").experiment_id
            ],
            filter_string=f"attributes.start_time >= '{start_date.isoformat()}'",
        )

        if not runs.empty:
            if task == "generation":
                metrics[task]["loss"] = runs["metrics.eval_loss"].tolist()
            else:
                metrics[task]["f1"] = runs["metrics.eval_f1"].tolist()
                metrics[task]["accuracy"] = runs["metrics.eval_accuracy"].tolist()

            metrics[task]["timestamp"] = pd.to_datetime(runs["start_time"]).tolist()

    return metrics


def generate_drift_report():
    """Generate data drift report using Evidently."""
    # Load reference and current data
    import psycopg2
    import pandas as pd

    conn_string = os.getenv(
        "DATABASE_URL", "postgresql://mluser:mlpassword@localhost:5432/mlpipeline"
    )

    # Get reference data (older)
    reference_query = """
    SELECT label, text
    FROM ag_news
    WHERE processed_at < NOW() - INTERVAL '7 days'
    LIMIT 1000
    """

    # Get current data
    current_query = """
    SELECT label, text
    FROM ag_news
    WHERE processed_at >= NOW() - INTERVAL '7 days'
    LIMIT 1000
    """

    reference_data = pd.read_sql(reference_query, conn_string)
    current_data = pd.read_sql(current_query, conn_string)

    # Create Evidently report
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])

    report.run(reference_data=reference_data, current_data=current_data)
    return report


def main():
    st.title("AG News ML Pipeline Dashboard")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Monitoring", "Admin"])

    if page == "Monitoring":
        st.header("📊 Model Monitoring")

        # MLflow metrics
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)

        metrics = get_mlflow_metrics()

        with col1:
            st.write("Generation Model Loss")
            if metrics["generation"]["loss"]:
                df = pd.DataFrame(
                    {
                        "timestamp": metrics["generation"]["timestamp"],
                        "loss": metrics["generation"]["loss"],
                    }
                )
                fig = px.line(
                    df, x="timestamp", y="loss", title="Generation Loss Over Time"
                )
                st.plotly_chart(fig)
            else:
                st.info("No generation metrics available for the last 7 days")

        with col2:
            st.write("Classification Model Metrics")
            if metrics["classification"]["f1"]:
                df = pd.DataFrame(
                    {
                        "timestamp": metrics["classification"]["timestamp"],
                        "F1 Score": metrics["classification"]["f1"],
                        "Accuracy": metrics["classification"]["accuracy"],
                    }
                )
                fig = px.line(
                    df,
                    x="timestamp",
                    y=["F1 Score", "Accuracy"],
                    title="Classification Metrics Over Time",
                )
                st.plotly_chart(fig)
            else:
                st.info("No classification metrics available for the last 7 days")

        # Data Drift Report
        st.subheader("Data Drift Analysis")
        if st.button("Generate Drift Report"):
            with st.spinner("Generating drift report..."):
                report = generate_drift_report()
                st.components.v1.html(report.get_html(), height=800)

    else:  # Admin page
        st.header("⚙️ Admin Controls")

        # Model Inference
        st.subheader("Test Model Inference")

        task = st.selectbox("Select Task", ["generation", "classification"])
        text = st.text_area("Enter Text for Inference")

        if st.button("Run Inference"):
            if text:
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/infer", json={"text": text, "task": task}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Result: {result['result']}")
                        if result.get("confidence"):
                            st.info(f"Confidence: {result['confidence']:.2f}")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
            else:
                st.warning("Please enter text for inference")

        # Model Retraining
        st.subheader("Model Retraining")

        retrain_task = st.selectbox(
            "Select Model to Retrain", ["generation", "classification"]
        )

        if st.button("Trigger Retraining"):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/retrain",
                    json={"task": retrain_task},
                    headers={"X-API-Key": API_KEY},
                )
                if response.status_code == 200:
                    st.success("Retraining initiated successfully!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

        # Model Registry Status
        st.subheader("Model Registry Status")

        try:
            for task in ["generation", "classification"]:
                model_name = f"lora_{task}"
                latest_version = mlflow.tracking.MlflowClient().get_latest_versions(
                    model_name, stages=["Production"]
                )

                if latest_version:
                    version = latest_version[0]
                    st.write(f"**{model_name}**")
                    st.write(f"- Version: {version.version}")
                    st.write(f"- Status: {version.current_stage}")
                    st.write(f"- Created: {version.creation_timestamp}")
                else:
                    st.info(f"No production version found for {model_name}")
        except Exception as e:
            st.error(f"Error fetching model registry information: {str(e)}")


if __name__ == "__main__":
    main()
