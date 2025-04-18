from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import mlflow
import torch
from transformers import AutoTokenizer
import os
from typing import List, Optional
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="AG News ML Pipeline API")

# Security
API_KEY = os.getenv("API_KEY", "your-api-key-here")
api_key_header = APIKeyHeader(name="X-API-Key")


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid API Key")


# Pydantic models
class InferenceRequest(BaseModel):
    text: str
    task: str  # "generation" or "classification"


class InferenceResponse(BaseModel):
    task: str
    result: str
    confidence: Optional[float] = None


class RetrainRequest(BaseModel):
    task: str  # "generation" or "classification"


class RetrainResponse(BaseModel):
    status: str
    message: str


# Load models
def load_model(task: str):
    if task not in ["generation", "classification"]:
        raise ValueError("Invalid task")

    model_name = f"lora_{task}"
    try:
        model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    try:
        model = load_model(request.task)

        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            if request.task == "generation":
                outputs = model.generate(
                    **inputs, max_length=100, num_return_sequences=1, temperature=0.7
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return InferenceResponse(
                    task=request.task, result=generated_text, confidence=None
                )
            else:  # classification
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0][predicted_class].item()

                # Map class index to label
                label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
                predicted_label = label_map[predicted_class]

                return InferenceResponse(
                    task=request.task, result=predicted_label, confidence=confidence
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest, api_key: str = Depends(get_api_key)):
    try:
        from app.models.train import ModelTrainer

        trainer = ModelTrainer()

        if request.task == "generation":
            trainer.train_generation()
        else:
            trainer.train_classification()

        return RetrainResponse(
            status="success", message=f"Successfully retrained {request.task} model"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
