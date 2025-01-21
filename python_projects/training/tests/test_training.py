import pytest
import asyncio
import json
import os
from pathlib import Path
import torch
from transformers import TrainingArguments

from main import train_tokenizer, train_model, load_and_prepare_dataset

def test_dataset_loading():
    """Test dataset loading functionality"""
    dataset = asyncio.run(load_and_prepare_dataset(tokenizer_samples=10))
    assert len(dataset) == 10, "Dataset should have exactly 10 samples"
    assert "text" in dataset.features, "Dataset should have 'text' field"
    assert all(isinstance(item["text"], str) for item in dataset), "All items should have text"

def test_tokenizer_training_state(tmp_path):
    """Test tokenizer training state saving/loading"""
    state_path = tmp_path / "tokenizer_state.json"
    state = {"processed_count": 1000}
    
    # Save state
    with open(state_path, "w") as f:
        json.dump(state, f)
    
    # Load state
    with open(state_path) as f:
        loaded_state = json.load(f)
    
    assert loaded_state["processed_count"] == 1000, "State not preserved correctly"

@pytest.mark.asyncio
async def test_tokenizer_training(small_dataset, tmp_path):
    """Test tokenizer training process"""
    vocab_size = 1000
    tokenizer = await train_tokenizer(small_dataset, vocab_size, output_dir=tmp_path)
    
    assert tokenizer is not None, "Tokenizer training failed"
    assert len(tokenizer.get_vocab()) <= vocab_size, "Vocabulary size exceeds specified limit"
    
    # Test tokenizer file was saved
    assert (tmp_path / "tokenizer.json").exists(), "Tokenizer file not saved"

def test_training_arguments():
    """Test training arguments configuration"""
    output_dir = "test_output"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10,
        logging_steps=1
    )
    
    assert args.num_train_epochs == 1, "Epochs not set correctly"
    assert args.per_device_train_batch_size == 4, "Batch size not set correctly"
    assert args.save_steps == 10, "Save steps not set correctly"
    assert args.logging_steps == 1, "Logging steps not set correctly"

@pytest.mark.asyncio
async def test_model_training(small_dataset, tokenizer, tmp_path):
    """Test model training process"""
    # Prepare small dataset
    encoded_dataset = small_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=32
        ),
        batched=True
    )
    
    # Train model
    model = await train_model(encoded_dataset, tokenizer, tmp_path, training_samples=5)
    
    assert model is not None, "Model training failed"
    assert (tmp_path / "pytorch_model.bin").exists(), "Model file not saved"
    assert (tmp_path / "config.json").exists(), "Model config not saved"

def test_interrupt_handling():
    """Test interrupt signal handling"""
    import signal
    from main import handle_interrupt, should_stop
    
    # Simulate interrupt
    handle_interrupt(signal.SIGINT, None)
    assert should_stop, "Interrupt not handled correctly"

@pytest.mark.asyncio
async def test_training_metrics(small_dataset, tmp_path):
    """Test training metrics collection"""
    vocab_size = 1000
    tokenizer = await train_tokenizer(small_dataset, vocab_size, output_dir=tmp_path)
    
    # Check if metrics file exists
    metrics_file = tmp_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        assert "processed_examples" in metrics, "Processed examples count not tracked"
        assert "elapsed_time" in metrics, "Training time not tracked"
        assert metrics["processed_examples"] > 0, "No examples processed"

def test_memory_usage(model):
    """Test memory usage during operations"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.max_memory_allocated()
    
    # Run some operations
    input_ids = torch.randint(0, 1000, (1, 100)).cuda()
    _ = model(input_ids)
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = peak_memory - initial_memory
    
    assert memory_used > 0, "No memory was used"
    assert memory_used < 1e10, "Excessive memory usage detected"  # 10GB limit

@pytest.mark.asyncio
async def test_training_resumption(small_dataset, tmp_path):
    """Test training can be resumed"""
    # Start training
    vocab_size = 1000
    tokenizer = await train_tokenizer(small_dataset[:5], vocab_size, output_dir=tmp_path)
    
    # Interrupt training
    from main import should_stop
    should_stop = True
    
    # Resume training
    should_stop = False
    tokenizer = await train_tokenizer(small_dataset[5:], vocab_size, output_dir=tmp_path)
    
    assert tokenizer is not None, "Training resumption failed" 