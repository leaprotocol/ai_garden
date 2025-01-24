import pytest
import asyncio
from pathlib import Path
from training_next.tokenizer_trainer import train_tokenizer, create_tokenizer

@pytest.mark.asyncio
async def test_tokenizer_training(small_dataset, tmp_path):
    """Test tokenizer training process"""
    vocab_size = 1000
    tokenizer = await train_tokenizer(small_dataset, vocab_size, output_dir=tmp_path)
    
    assert tokenizer is not None, "Tokenizer training failed"
    assert len(tokenizer.get_vocab()) <= vocab_size, "Vocabulary size exceeds specified limit"
    
    # Test tokenizer file was saved
    assert (tmp_path / "tokenizer.json").exists(), "Tokenizer file not saved"

@pytest.mark.asyncio
async def test_tokenizer_loading(small_dataset, tmp_path):
    """Test tokenizer loading from saved file"""
    vocab_size = 1000
    tokenizer = await train_tokenizer(small_dataset, vocab_size, output_dir=tmp_path)
    
    # Try loading the saved tokenizer
    loaded_tokenizer = create_tokenizer(tmp_path)
    assert loaded_tokenizer is not None, "Failed to load saved tokenizer"
    assert len(loaded_tokenizer.get_vocab()) == len(tokenizer.get_vocab()), "Vocabulary size mismatch"

def test_tokenizer_special_tokens(tokenizer):
    """Test special tokens in tokenizer"""
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|pad|>"
    }
    
    for token_type, token_value in special_tokens.items():
        assert hasattr(tokenizer, token_type), f"Missing {token_type}"
        assert getattr(tokenizer, token_type) == token_value, f"Incorrect {token_type}"

@pytest.mark.asyncio
async def test_tokenizer_encoding(tokenizer):
    """Test tokenizer encoding/decoding"""
    test_text = "Hello, world!"
    
    # Test encoding
    encoded = tokenizer.encode(test_text)
    assert isinstance(encoded, list), "Encoded output should be a list"
    assert len(encoded) > 0, "Encoded output should not be empty"
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str), "Decoded output should be a string"
    assert len(decoded) > 0, "Decoded output should not be empty"

@pytest.mark.asyncio
async def test_tokenizer_batch_encoding(tokenizer):
    """Test tokenizer batch encoding"""
    texts = ["Hello, world!", "Testing batch encoding", "Multiple texts"]
    
    # Test batch encoding
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    assert "input_ids" in encoded, "Missing input_ids in batch encoding"
    assert "attention_mask" in encoded, "Missing attention_mask in batch encoding"
    assert encoded.input_ids.shape[0] == len(texts), "Batch size mismatch"

@pytest.mark.asyncio
async def test_tokenizer_max_length(tokenizer):
    """Test tokenizer max length handling"""
    long_text = " ".join(["test"] * 1000)  # Very long text
    max_length = 128
    
    encoded = tokenizer(
        long_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    assert encoded.input_ids.shape[1] <= max_length, "Encoded sequence exceeds max_length" 