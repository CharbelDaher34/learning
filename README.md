# AG News LLM Project

This project implements a complete machine learning system for fine-tuning, deploying, and monitoring language models on the AG News dataset.

## Features

- **ETL Pipeline**: Downloads and processes the AG News dataset from Hugging Face
- **Fine-tuning LLMs**: Fine-tunes a base Hugging Face model with two LoRA adapters:
  - Text generation adapter
  - News classification adapter
- **Continual Learning**: Automatically retrains models when performance improves
- **MLflow Integration**: Tracks models, experiments, and metrics
- **FastAPI Backend**: Serves classification and generation endpoints
- **Streamlit Dashboard**: Monitors model performance and allows for interactive testing
- **Automatic Deployment**: Deploys new adapters when metrics exceed previous best

## Project Structure

```
├── etl/              # Data processing pipeline
├── models/           # Model training and inference
├── api/              # FastAPI backend
├── dashboard/        # Streamlit admin dashboard
├── mlflow/           # MLflow tracking data
├── scripts/          # Utility scripts
├── pyproject.toml    # Project configuration
└── README.md         # Project documentation
```

## Prerequisites

- Python 3.8+
- uv package manager
- CUDA-compatible GPU (recommended for training)

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ag-news-llm-project
```

2. Create and activate a virtual environment:

```bash
uv init
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
uv add fastapi uvicorn streamlit transformers bitsandbytes accelerate peft mlflow pydantic pandas scikit-learn typer datasets pytorch matplotlib seaborn plotly httpx
uv add pytest black flake8 --dev
```

## Usage

### ETL Pipeline

Download and process the AG News dataset:

```bash
uv run etl
```

### Training Models

Train the classification LoRA adapter:

```bash
uv run train-cls
```

Train the generation LoRA adapter:

```bash
uv run train-gen
```

### Running the API

Start the FastAPI backend:

```bash
uv run api
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

### Running the Dashboard

Start the Streamlit dashboard:

```bash
uv run dashboard
```

The dashboard will be available at http://localhost:8501.

### Monitoring and Retraining

Manually trigger monitoring and potential retraining:

```bash
# For classification model
uv run scripts.monitor_and_retrain --model_type classification

# For generation model
uv run scripts.monitor_and_retrain --model_type generation

# Force retraining regardless of metrics
uv run scripts.monitor_and_retrain --model_type classification --force_retrain
```

## API Endpoints

- `POST /classify`: Classify news text into categories
- `POST /generate`: Generate news text based on a prompt

## Dashboard Features

- **Monitor**: Track model metrics and performance over time
- **Retrain**: Manually trigger model evaluation and retraining
- **Test**: Interactive testing of deployed models

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The AG News dataset from Hugging Face Datasets
- The transformers library by Hugging Face
- PEFT (Parameter-Efficient Fine-Tuning) for LoRA adapters
