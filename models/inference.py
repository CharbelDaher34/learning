"""
Inference utilities for using the trained models.
"""

import os
import torch
from peft import PeftModel
from transformers import pipeline
from typing import Dict, Any, List, Optional, Union, Tuple
import mlflow
from mlflow.tracking import MlflowClient

from .base_model import get_model_and_tokenizer, get_model_info, DEFAULT_MODEL_NAME
from .mlflow_utils import (
    get_latest_model_version,
    setup_mlflow,
)
from etl.preprocess import AG_NEWS_CLASSES


class InferenceModel:
    """Base class for inference models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize the inference model.

        Args:
            model_name: Name of the base model
            adapter_path: Path to the LoRA adapter
            device: Device to use for inference
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device

        # Load base model and tokenizer
        self.base_model, self.tokenizer = get_model_and_tokenizer(
            model_name=model_name,
            device_map=device,
        )

        # Load adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            self._load_adapter(adapter_path)

    def _load_adapter(self, adapter_path: str):
        """
        Load a LoRA adapter.

        Args:
            adapter_path: Path to the adapter
        """
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, *args, **kwargs):
        """Make predictions with the model."""
        raise NotImplementedError("Subclasses must implement this method")


class ClassificationModel(InferenceModel):
    """Classification model for AG News categorization."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        num_classes: int = 4,
    ):
        """
        Initialize the classification model.

        Args:
            model_name: Name of the base model
            adapter_path: Path to the LoRA adapter
            device: Device to use for inference
            num_classes: Number of classes
        """
        super().__init__(model_name, adapter_path, device)
        self.num_classes = num_classes

        # If no adapter path is provided, try to initialize without adapter
        if not adapter_path:
            # Initialize classifier
            self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, num_classes).to(
                self.base_model.device
            )

    def _load_adapter(self, adapter_path: str):
        """
        Load a LoRA adapter and classifier for classification.

        Args:
            adapter_path: Path to the adapter
        """
        # Load the LoRA adapter
        self.base_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=False,
        )

        # Load the classifier
        classifier_path = os.path.join(adapter_path, "classifier.pt")
        if os.path.exists(classifier_path):
            self.classifier = torch.nn.Linear(
                self.base_model.config.hidden_size, self.num_classes
            ).to(self.base_model.device)
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=self.base_model.device)
            )
        else:
            raise ValueError(f"Classifier weights not found at {classifier_path}")

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Classify text into AG News categories.

        Args:
            texts: Text or list of texts to classify
            batch_size: Batch size for inference

        Returns:
            List of dictionaries with class predictions and probabilities
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(self.base_model.device)

        # Process in batches
        results = []
        for i in range(0, len(texts), batch_size):
            batch_inputs = {k: v[i : i + batch_size] for k, v in inputs.items()}

            # Get the last hidden state from base model
            with torch.no_grad():
                outputs = self.base_model(
                    **batch_inputs,
                    output_hidden_states=True,
                )
                last_hidden_state = outputs.hidden_states[-1]
                cls_output = last_hidden_state[:, 0, :]

                # Pass through the classifier
                logits = self.classifier(cls_output)
                probabilities = torch.softmax(logits, dim=1)

            # Convert to numpy for easier processing
            probabilities = probabilities.cpu().numpy()
            predictions = probabilities.argmax(axis=1)

            # Format results
            for j, (pred, probs) in enumerate(zip(predictions, probabilities)):
                # Convert to original label index (1-4)
                label_idx = int(pred) + 1

                results.append(
                    {
                        "label": label_idx,
                        "class_name": AG_NEWS_CLASSES[label_idx],
                        "probabilities": {
                            AG_NEWS_CLASSES[k + 1]: float(probs[k]) for k in range(len(probs))
                        },
                    }
                )

        return results


class GenerationModel(InferenceModel):
    """Text generation model fine-tuned on AG News."""

    def _load_adapter(self, adapter_path: str):
        """
        Load a LoRA adapter for generation.

        Args:
            adapter_path: Path to the adapter
        """
        # Load the LoRA adapter
        self.base_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=False,
        )

    def predict(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate text based on prompts.

        Args:
            prompts: Text prompt or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return

        Returns:
            List of dictionaries with generated text
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Create a pipeline for easier text generation
        text_generator = pipeline(
            "text-generation",
            model=self.base_model,
            tokenizer=self.tokenizer,
            device=self.base_model.device,
        )

        # Generate text for each prompt
        results = []
        for prompt in prompts:
            outputs = text_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

            # Process outputs
            prompt_results = []
            for output in outputs:
                generated_text = output["generated_text"]

                # Make sure to return only the newly generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt) :].strip()

                prompt_results.append(
                    {
                        "generated_text": generated_text,
                        "full_text": output["generated_text"],
                    }
                )

            # For a single prompt with a single sequence, return just the result
            if len(prompt_results) == 1:
                results.append(prompt_results[0])
            else:
                results.append(prompt_results)

        # For a single input prompt, return just the result
        if len(results) == 1:
            return results[0]

        return results


def load_classification_model_from_mlflow(
    model_name: str = "lora-cls",
    stage: str = "Production",
    base_model_name: str = DEFAULT_MODEL_NAME,
) -> ClassificationModel:
    """
    Load a classification model from MLflow.

    Args:
        model_name: Name of the registered model
        stage: Model stage to load
        base_model_name: Name of the base model

    Returns:
        Loaded classification model
    """
    # Setup MLflow
    setup_mlflow()
    client = MlflowClient()

    # Get latest version
    version = get_latest_model_version(model_name, stage)
    if not version:
        raise ValueError(f"No model found for {model_name} in stage {stage}")

    # Get the model URI
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id

    # Download the adapter from MLflow
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="adapter",
    )

    # Load the model
    return ClassificationModel(
        model_name=base_model_name,
        adapter_path=local_path,
    )


def load_generation_model_from_mlflow(
    model_name: str = "lora-gen",
    stage: str = "Production",
    base_model_name: str = DEFAULT_MODEL_NAME,
) -> GenerationModel:
    """
    Load a generation model from MLflow.

    Args:
        model_name: Name of the registered model
        stage: Model stage to load
        base_model_name: Name of the base model

    Returns:
        Loaded generation model
    """
    # Setup MLflow
    setup_mlflow()
    client = MlflowClient()

    # Get latest version
    version = get_latest_model_version(model_name, stage)
    if not version:
        raise ValueError(f"No model found for {model_name} in stage {stage}")

    # Get the model URI
    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id

    # Download the adapter from MLflow
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="adapter",
    )

    # Load the model
    return GenerationModel(
        model_name=base_model_name,
        adapter_path=local_path,
    )


def get_category_examples(category: str) -> str:
    """
    Get example prompts for a news category.

    Args:
        category: News category

    Returns:
        Example prompt for the category
    """
    examples = {
        "World": "Write a news article about international diplomatic talks.",
        "Sports": "Write a news article about a major sports tournament.",
        "Business": "Write a news article about stock market trends.",
        "Sci/Tech": "Write a news article about recent advances in renewable energy.",
    }

    return examples.get(category, "Write a news article.")
