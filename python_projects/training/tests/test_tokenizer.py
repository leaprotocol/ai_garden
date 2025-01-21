import pytest
import json
from pathlib import Path
import torch

def test_tokenizer_exists(model_path):
    """Test that tokenizer file exists"""
    tokenizer_path = model_path / "tokenizer.json"
    assert tokenizer_path.exists(), "Tokenizer file not found"

def test_tokenizer_loads(tokenizer):
    """Test that tokenizer can be loaded"""
    assert tokenizer is not None, "Tokenizer failed to load"

def test_special_tokens(tokenizer):
    """Test special tokens are properly set"""
    assert tokenizer.pad_token == "<|pad|>"
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.bos_token == "<|endoftext|>"
    assert tokenizer.unk_token == "<|endoftext|>"

def test_tokenize_decode_roundtrip(tokenizer, test_data):
    """Test that tokenization followed by decoding returns original text"""
    for text in test_data:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert isinstance(tokens, list), "Encoded tokens should be a list"
        assert all(isinstance(t, int) for t in tokens), "All tokens should be integers"
        assert decoded.strip() == text.strip(), "Roundtrip text does not match"

def test_batch_tokenization(tokenizer, test_data):
    """Test batch tokenization"""
    encoded = tokenizer(test_data, padding=True, truncation=True, return_tensors="pt")
    assert isinstance(encoded.input_ids, torch.Tensor)
    assert len(encoded.input_ids) == len(test_data)

def test_tokenizer_vocabulary(tokenizer):
    """Test tokenizer vocabulary properties"""
    vocab = tokenizer.get_vocab()
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert "<|pad|>" in vocab, "Pad token not in vocabulary"
    assert "<|endoftext|>" in vocab, "EOS token not in vocabulary"

def test_tokenizer_max_length(tokenizer):
    """Test tokenizer handles long sequences"""
    long_text = "a" * 1000
    tokens = tokenizer.encode(long_text, truncation=True, max_length=100)
    assert len(tokens) <= 100, "Truncation failed"

def test_tokenizer_empty_input(tokenizer):
    """Test tokenizer handles empty input"""
    tokens = tokenizer.encode("")
    assert isinstance(tokens, list), "Should return empty list for empty input"

def test_tokenizer_whitespace(tokenizer):
    """Test tokenizer handles whitespace"""
    text = "  Hello   World  "
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == text.strip(), "Whitespace handling incorrect"

def test_tokenizer_special_chars(tokenizer):
    """Test tokenizer handles special characters"""
    special_text = "Hello! @#$%^&* World\n\t123"
    tokens = tokenizer.encode(special_text)
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == special_text.strip(), "Special characters not preserved"

def test_tokenizer_save_load(tokenizer, tmp_path):
    """Test tokenizer can be saved and loaded"""
    # Save
    save_path = tmp_path / "test_tokenizer"
    tokenizer.save_pretrained(save_path)
    
    # Load
    loaded_tokenizer = type(tokenizer).from_pretrained(save_path)
    
    # Compare
    assert tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
    assert tokenizer.all_special_tokens == loaded_tokenizer.all_special_tokens 