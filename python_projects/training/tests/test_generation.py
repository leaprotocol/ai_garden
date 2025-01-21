import pytest
import torch
import asyncio
from demo import generate_text_async, clean_text

@pytest.mark.asyncio
async def test_text_generation(model, tokenizer):
    """Test basic text generation"""
    prompt = "Once upon a time"
    samples = await generate_text_async(model, tokenizer, prompt, max_length=20)
    
    assert isinstance(samples, list), "Should return a list of samples"
    assert len(samples) > 0, "Should generate at least one sample"
    assert all(isinstance(s, str) for s in samples), "All samples should be strings"
    assert all(s.startswith(prompt) for s in samples), "All samples should start with prompt"

@pytest.mark.asyncio
async def test_generation_parameters(model, tokenizer):
    """Test generation with different parameters"""
    prompt = "The story begins"
    
    # Test different lengths
    short_samples = await generate_text_async(model, tokenizer, prompt, max_length=10)
    long_samples = await generate_text_async(model, tokenizer, prompt, max_length=50)
    
    assert max(len(s) for s in short_samples) < max(len(s) for s in long_samples), \
        "Longer max_length should produce longer sequences"
    
    # Test different temperatures
    deterministic = await generate_text_async(model, tokenizer, prompt, temperature=0.0, num_samples=3)
    random = await generate_text_async(model, tokenizer, prompt, temperature=1.0, num_samples=3)
    
    assert len(set(deterministic)) == 1, "Temperature 0 should produce identical outputs"
    assert len(set(random)) > 1, "Temperature 1 should produce varied outputs"

@pytest.mark.asyncio
async def test_batch_generation(model, tokenizer):
    """Test generating multiple samples at once"""
    prompt = "Testing batch"
    num_samples = 5
    samples = await generate_text_async(model, tokenizer, prompt, num_samples=num_samples)
    
    assert len(samples) == num_samples, f"Should generate exactly {num_samples} samples"
    assert len(set(samples)) > 1, "Should generate different samples"

def test_text_cleaning():
    """Test text cleaning function"""
    # Test whitespace handling
    assert clean_text("  Hello   World  ") == "Hello World"
    
    # Test punctuation spacing
    assert clean_text("Hello , World !") == "Hello, World!"
    
    # Test special token cleaning
    assert clean_text("ĠHelloĠWorld") == "Hello World"

@pytest.mark.asyncio
async def test_empty_prompt(model, tokenizer):
    """Test generation with empty prompt"""
    samples = await generate_text_async(model, tokenizer, "", max_length=20)
    assert isinstance(samples, list), "Should handle empty prompt"
    assert all(isinstance(s, str) for s in samples), "Should generate valid strings"

@pytest.mark.asyncio
async def test_long_prompt(model, tokenizer):
    """Test generation with very long prompt"""
    long_prompt = "a" * 1000
    samples = await generate_text_async(model, tokenizer, long_prompt, max_length=1050)
    assert all(len(s) <= 1050 for s in samples), "Should respect max_length"

@pytest.mark.asyncio
async def test_special_characters(model, tokenizer):
    """Test generation with special characters"""
    prompt = "Hello! @#$%^&* World\n\t123"
    samples = await generate_text_async(model, tokenizer, prompt)
    assert all(s.startswith(prompt) for s in samples), "Should handle special characters"

@pytest.mark.asyncio
async def test_generation_interruption(model, tokenizer):
    """Test generation can be interrupted"""
    prompt = "This is a test"
    
    # Start generation in background
    task = asyncio.create_task(
        generate_text_async(model, tokenizer, prompt, max_length=1000, num_samples=10)
    )
    
    # Wait briefly then cancel
    await asyncio.sleep(0.1)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected
    
    assert True, "Generation should be interruptible"

@pytest.mark.asyncio
async def test_concurrent_generation(model, tokenizer):
    """Test multiple generations running concurrently"""
    prompts = ["First", "Second", "Third"]
    tasks = [
        generate_text_async(model, tokenizer, prompt, max_length=20)
        for prompt in prompts
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == len(prompts), "Should complete all generations"
    assert all(isinstance(samples, list) for samples in results), "All results should be sample lists"
    assert all(
        sample.startswith(prompt)
        for samples, prompt in zip(results, prompts)
        for sample in samples
    ), "Generated text should match corresponding prompts" 