"""
FastAPI backend for AG News classification and text generation.
"""

import os
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any, Optional, List

from models import (
    load_classification_model_from_mlflow,
    load_generation_model_from_mlflow,
    get_latest_model_version,
    get_category_examples,
)
from .models import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    GenerationRequest,
    GenerationResponse,
    GenerationResult,
    HealthResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AG News API",
    description="API for classifying and generating news texts",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of models
classification_model = None
generation_model = None
model_versions = {}


def get_classification_model():
    """Get or initialize the classification model."""
    global classification_model, model_versions

    if classification_model is None:
        logger.info("Loading classification model...")
        try:
            classification_model = load_classification_model_from_mlflow()
            model_versions["classification"] = get_latest_model_version("lora-cls") or "unknown"
            logger.info(
                f"Classification model loaded (version: {model_versions['classification']})"
            )
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Classification model failed to load: {str(e)}",
            )

    return classification_model


def get_generation_model():
    """Get or initialize the generation model."""
    global generation_model, model_versions

    if generation_model is None:
        logger.info("Loading generation model...")
        try:
            generation_model = load_generation_model_from_mlflow()
            model_versions["generation"] = get_latest_model_version("lora-gen") or "unknown"
            logger.info(f"Generation model loaded (version: {model_versions['generation']})")
        except Exception as e:
            logger.error(f"Error loading generation model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation model failed to load: {str(e)}",
            )

    return generation_model


@app.get("/", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "models": model_versions,
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify(
    request: ClassificationRequest,
    model=Depends(get_classification_model),
):
    """
    Classify a text into an AG News category.

    This endpoint uses a fine-tuned LoRA adapter to classify news text into one of:
    - World
    - Sports
    - Business
    - Science/Technology
    """
    start_time = time.time()

    try:
        # Perform classification
        results = model.predict(request.text)

        # Return results
        return ClassificationResponse(
            results=results,
            model_version=model_versions.get("classification", "unknown"),
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}",
        )


@app.post("/generate", response_model=GenerationResponse)
async def generate(
    request: GenerationRequest,
    model=Depends(get_generation_model),
):
    """
    Generate text based on a prompt.

    This endpoint uses a fine-tuned LoRA adapter to generate news text.
    You can optionally specify a category to guide the generation.
    """
    start_time = time.time()

    try:
        # Prepare the prompt
        prompt = request.prompt

        # If a category is provided, add it to the prompt
        if request.category:
            category_example = get_category_examples(request.category)
            # Combine the category example with the user prompt
            if category_example and not prompt.startswith(category_example):
                if not prompt.endswith("."):
                    prompt = prompt + "."
                prompt = f"{category_example} {prompt}"

        # Perform generation
        results = model.predict(
            prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences,
        )

        # Return results
        return GenerationResponse(
            results=results,
            model_version=model_versions.get("generation", "unknown"),
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


def start():
    """Start the API server."""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    start()
