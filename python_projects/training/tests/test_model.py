import pytest
import torch
import numpy as np
from pathlib import Path

def test_model_exists(model_path):
    """Test that model files exist"""
    config_path = model_path / "config.json"
    model_path = model_path / "pytorch_model.bin"
    assert config_path.exists(), "Model config not found"
    assert model_path.exists(), "Model weights not found"

def test_model_loads(model):
    """Test that model can be loaded"""
    assert model is not None, "Model failed to load"

def test_model_generation(model, tokenizer):
    """Test basic text generation"""
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    assert output.shape[1] > input_ids.shape[1], "Model should generate longer sequence than input"
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(generated_text) > len(prompt), "Generated text should be longer than prompt"
    assert generated_text.startswith(prompt), "Generated text should start with prompt"

def test_model_batch_generation(model, tokenizer):
    """Test batch text generation"""
    prompts = ["Hello", "Testing", "Once upon"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    assert len(outputs) == len(prompts), "Should generate one sequence per prompt"
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    assert all(len(text) > 0 for text in generated_texts), "All generated texts should be non-empty"

def test_model_temperature(model, tokenizer):
    """Test generation with different temperatures"""
    prompt = "The quick brown"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate with temperature = 0 (deterministic)
    with torch.no_grad():
        outputs_deterministic = [
            model.generate(
                input_ids,
                max_length=20,
                temperature=0.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            ) for _ in range(3)
        ]
    
    # All outputs should be identical with temperature = 0
    texts_deterministic = [tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs_deterministic]
    assert all(text == texts_deterministic[0] for text in texts_deterministic), "Deterministic generation not consistent"
    
    # Generate with high temperature (more random)
    with torch.no_grad():
        outputs_random = [
            model.generate(
                input_ids,
                max_length=20,
                temperature=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            ) for _ in range(3)
        ]
    
    texts_random = [tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs_random]
    # High temperature should produce different outputs
    assert len(set(texts_random)) > 1, "High temperature should produce varied outputs"

def test_model_save_load(model, tmp_path):
    """Test model can be saved and loaded"""
    # Save
    save_path = tmp_path / "test_model"
    model.save_pretrained(save_path)
    
    # Load
    loaded_model = type(model).from_pretrained(save_path)
    
    # Compare model parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2), "Model parameters don't match after save/load"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_to_gpu(model):
    """Test model can be moved to GPU"""
    model.to("cuda")
    assert next(model.parameters()).is_cuda, "Model not moved to GPU"

def test_model_gradient_flow(model, tokenizer):
    """Test model gradient computation"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Forward pass
    input_text = "Testing gradient flow"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs, labels=inputs.input_ids)
    
    # Check loss
    assert isinstance(outputs.loss, torch.Tensor), "Model should return loss"
    assert not torch.isnan(outputs.loss), "Loss should not be NaN"
    
    # Backward pass
    outputs.loss.backward()
    
    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.requires_grad:
            has_grad = True
            assert param.grad is not None, "Parameters should have gradients"
            assert not torch.isnan(param.grad).any(), "Gradients should not be NaN"
    
    assert has_grad, "Model should have at least one parameter with gradients" 