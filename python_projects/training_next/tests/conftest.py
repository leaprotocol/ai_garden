import pytest
import asyncio
import torch
from pathlib import Path
from training_next.dataset_utils import load_and_prepare_dataset
from training_next.tokenizer_trainer import train_tokenizer

@pytest.fixture
def small_dataset():
    """Fixture providing a small dataset for testing"""
    return asyncio.run(load_and_prepare_dataset(tokenizer_samples=10))

@pytest.fixture
def tmp_path():
    """Fixture providing a temporary directory path"""
    path = Path("test_output")
    path.mkdir(exist_ok=True)
    yield path
    # Cleanup after tests
    if path.exists():
        import shutil
        shutil.rmtree(path)

@pytest.fixture
async def tokenizer(small_dataset, tmp_path):
    """Fixture providing a trained tokenizer"""
    return await train_tokenizer(small_dataset, vocab_size=1000, output_dir=tmp_path)

@pytest.fixture
def device():
    """Fixture providing the compute device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 