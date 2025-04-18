"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class NewsCategory(str, Enum):
    """Valid news categories."""

    WORLD = "World"
    SPORTS = "Sports"
    BUSINESS = "Business"
    SCITECH = "Sci/Tech"


class ClassificationRequest(BaseModel):
    """Request model for text classification."""

    text: str = Field(..., description="The text to classify", min_length=1)


class ClassificationResult(BaseModel):
    """Single classification result."""

    label: int = Field(..., description="The predicted label (1-4)")
    class_name: str = Field(..., description="The class name")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class ClassificationResponse(BaseModel):
    """Response model for text classification."""

    results: List[ClassificationResult] = Field(..., description="Classification results")
    model_version: str = Field(..., description="The model version used for inference")
    processing_time: float = Field(..., description="Time taken for processing in seconds")


class GenerationRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="The prompt for text generation", min_length=1)
    category: Optional[NewsCategory] = Field(
        None, description="Optional news category to guide generation"
    )
    max_new_tokens: Optional[int] = Field(100, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(
        0.7, description="Temperature for sampling", ge=0.1, le=2.0
    )
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1)
    num_return_sequences: Optional[int] = Field(
        1, description="Number of sequences to return", ge=1, le=5
    )

    @validator("max_new_tokens")
    def validate_max_tokens(cls, v):
        if v is not None and (v < 10 or v > 500):
            raise ValueError("max_new_tokens must be between 10 and 500")
        return v


class GenerationResult(BaseModel):
    """Single generation result."""

    generated_text: str = Field(..., description="The generated text")
    full_text: Optional[str] = Field(None, description="The full text including the prompt")


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    results: Union[GenerationResult, List[GenerationResult]] = Field(
        ..., description="Generation results"
    )
    model_version: str = Field(..., description="The model version used for inference")
    processing_time: float = Field(..., description="Time taken for processing in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    models: Dict[str, str] = Field(..., description="Available models and their versions")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    type: str = Field(..., description="Error type")
