# AG News ML Pipeline

A complete end-to-end machine learning pipeline for processing and analyzing AG News dataset using LoRA fine-tuning, MLflow tracking, and real-time monitoring.

## Features

- ETL pipeline using Metaflow for data processing
- LoRA fine-tuning for both text generation and classification
- MLflow experiment tracking and model registry
- FastAPI backend for model serving
- Streamlit dashboard for monitoring and administration
- Docker-based deployment with docker-compose
- Continuous monitoring with Evidently

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── etl/
│   │   └── data_pipeline.py # Metaflow ETL pipeline
│   ├── models/
│   │   └── train.py         # LoRA training code
│   └── dashboard.py         # Streamlit dashboard
├── tests/
│   └── ...                  # Test files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10
- Docker and Docker Compose
- PostgreSQL
- CUDA-compatible GPU (recommended for training)

## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd ag-news-pipeline
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export DATABASE_URL="postgresql://mluser:mlpassword@localhost:5432/mlpipeline"
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   export API_KEY="your-api-key-here"
   ```

## Running with Docker Compose

1. Build and start the services:

   ```bash
   docker-compose up --build
   ```

2. Access the services:
   - FastAPI Swagger UI: http://localhost:8000/docs
   - Streamlit Dashboard: http://localhost:8501
   - MLflow UI: http://localhost:5000

## Running Components Individually

### ETL Pipeline

```bash
python -m app.etl.data_pipeline.py
```

### Model Training

```bash
python -m app.models.train
```

### FastAPI Backend

```bash
uvicorn app.api.main:app --reload
```

### Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

## API Endpoints

### /infer

- POST request for model inference
- Requires text input and task type (generation/classification)
- Returns generated text or classification with confidence

### /retrain

- POST request to trigger model retraining
- Requires API key authentication
- Returns training status

## Monitoring

The Streamlit dashboard provides:

- Model performance metrics
- Data drift analysis
- Model registry status
- Interactive inference testing
- Retraining triggers

## Testing

Run the test suite:

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
