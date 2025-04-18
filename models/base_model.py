"""
Base model loader for Hugging Face text-generation models.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Dict, Any, Optional

DEFAULT_MODEL_NAME = "gpt2"  # Can be changed to other models like "facebook/opt-350m"
MODEL_MAX_LENGTH = 512


def get_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    use_4bit: bool = True,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained language model and tokenizer from Hugging Face.

    Args:
        model_name: Name of the Hugging Face model to load
        use_4bit: Whether to use 4-bit quantization for memory efficiency
        device_map: Device mapping strategy for model loading
        cache_dir: Directory to cache models

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    # Configure quantization settings if enabled
    if use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    # Load model with quantization if available
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_info(model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """
    Get information about the model for tracking purposes.

    Args:
        model_name: Name of the Hugging Face model

    Returns:
        Dictionary with model information
    """
    return {
        "model_name": model_name,
        "type": "causal_lm",
        "source": "huggingface",
    }


if __name__ == "__main__":
    # Test loading the model
    model, tokenizer = get_model_and_tokenizer()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Test tokenization and generation
    inputs = tokenizer("This is a test sentence for text generation.", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
