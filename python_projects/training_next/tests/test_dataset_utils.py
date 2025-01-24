import pytest
import asyncio
from training_next.dataset_utils import load_and_prepare_dataset

@pytest.mark.asyncio
async def test_dataset_loading():
    """Test dataset loading functionality"""
    dataset = await load_and_prepare_dataset(tokenizer_samples=10)
    assert len(dataset) == 10, "Dataset should have exactly 10 samples"
    assert "text" in dataset.features, "Dataset should have 'text' field"
    assert all(isinstance(item["text"], str) for item in dataset), "All items should have text"

@pytest.mark.asyncio
async def test_dataset_content():
    """Test dataset content quality"""
    dataset = await load_and_prepare_dataset(tokenizer_samples=5)
    
    for item in dataset:
        text = item["text"]
        assert len(text.strip()) > 0, "Text should not be empty"
        assert text.strip() == text, "Text should be properly stripped"
        assert len(text.split()) >= 2, "Text should have at least 2 words"

@pytest.mark.asyncio
async def test_dataset_preprocessing():
    """Test dataset preprocessing steps"""
    dataset = await load_and_prepare_dataset(tokenizer_samples=5)
    
    for item in dataset:
        text = item["text"]
        # Check for common preprocessing issues
        assert not text.startswith(" "), "Text should not start with whitespace"
        assert not text.endswith(" "), "Text should not end with whitespace"
        assert "\n\n" not in text, "Text should not have multiple consecutive newlines"

@pytest.mark.asyncio
async def test_dataset_size_limit():
    """Test dataset size limiting"""
    small_size = 5
    large_size = 1000
    
    small_dataset = await load_and_prepare_dataset(tokenizer_samples=small_size)
    large_dataset = await load_and_prepare_dataset(tokenizer_samples=large_size)
    
    assert len(small_dataset) == small_size, "Small dataset size incorrect"
    assert len(large_dataset) == large_size, "Large dataset size incorrect"

@pytest.mark.asyncio
async def test_dataset_shuffling():
    """Test dataset shuffling"""
    # Load same dataset twice with different seeds
    dataset1 = await load_and_prepare_dataset(tokenizer_samples=100, seed=42)
    dataset2 = await load_and_prepare_dataset(tokenizer_samples=100, seed=43)
    
    # Convert to lists for comparison
    texts1 = [item["text"] for item in dataset1]
    texts2 = [item["text"] for item in dataset2]
    
    # Check that the datasets are different (shuffled)
    assert texts1 != texts2, "Datasets with different seeds should be shuffled differently"
    
    # Check that both datasets have the same content (just in different order)
    assert sorted(texts1) == sorted(texts2), "Datasets should have same content after sorting" 