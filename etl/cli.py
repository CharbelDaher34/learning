"""
CLI interface for ETL operations using Typer.
"""

import typer
from pathlib import Path
from .download_data import download_ag_news
from .preprocess import process_ag_news

app = typer.Typer(help="ETL operations for AG News dataset")


@app.command()
def download(output_dir: str = "etl/data"):
    """
    Download the AG News dataset from Hugging Face.
    """
    download_ag_news(output_dir)


@app.command()
def preprocess(data_dir: str = "etl/data"):
    """
    Preprocess the downloaded AG News dataset.
    """
    process_ag_news(data_dir)


@app.command()
def run_all(data_dir: str = "etl/data"):
    """
    Run the complete ETL pipeline (download + preprocess).
    """
    typer.echo("Starting ETL pipeline...")

    # Create directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download data
    typer.echo("\n=== Downloading AG News dataset ===")
    download_ag_news(data_dir)

    # Preprocess data
    typer.echo("\n=== Preprocessing AG News dataset ===")
    process_ag_news(data_dir)

    typer.echo("\n=== ETL pipeline completed successfully ===")


if __name__ == "__main__":
    app()
