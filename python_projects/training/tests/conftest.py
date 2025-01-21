import pytest
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

@pytest.fixture
def test_data():
    """Return a small test dataset"""
    return ["Hello world", "Testing is important", "This is a test sentence"]

@pytest.fixture
def model_path():
    """Return the path to model directory"""
    return Path("tiny_gpt")

@pytest.fixture
def tokenizer(model_path):
    """Load the tokenizer for testing"""
    try:
        return PreTrainedTokenizerFast.from_pretrained(str(model_path))
    except:
        pytest.skip("Tokenizer not found - run training first")

@pytest.fixture
def model(model_path):
    """Load the model for testing"""
    try:
        return AutoModelForCausalLM.from_pretrained(str(model_path))
    except:
        pytest.skip("Model not found - run training first")

@pytest.fixture
def small_dataset():
    """Load a small portion of the dataset"""
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train[:10]")
        return dataset
    except:
        pytest.skip("Could not load dataset - check internet connection")

@pytest.fixture
def device():
    """Return the compute device to use"""
    return "cuda" if torch.cuda.is_available() else "cpu" 