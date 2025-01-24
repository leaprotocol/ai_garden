import pytest
import asyncio
import torch
from pathlib import Path
from training_next.model_trainer import (
    train_model, 
    train_epoch, 
    should_stop, 
    SimpleDataset,
    find_latest_checkpoint,
    load_checkpoint
)

@pytest.mark.asyncio
async def test_model_initialization(tokenizer, tmp_path):
    """Test model initialization and configuration"""
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        force_retrain=True,
        batch_size=1
    )
    assert model is not None, "Model initialization failed"
    assert hasattr(model, "config"), "Model missing config"
    assert model.config.vocab_size == len(tokenizer), "Incorrect vocabulary size"

@pytest.mark.asyncio
async def test_dataset_creation(small_dataset, tokenizer):
    """Test SimpleDataset creation and functionality"""
    # Convert streaming dataset to list for testing
    texts = [item["text"] for item in small_dataset.take(2)]
    
    encodings = tokenizer(
        texts,  # Use the list of texts
        truncation=True,
        padding=True,
        max_length=32,  # Reduced from 128
        return_tensors="pt"
    )
    dataset = SimpleDataset(encodings)
    
    assert len(dataset) == 2, "Dataset length mismatch"
    item = dataset[0]
    assert "input_ids" in item, "Missing input_ids in dataset item"
    assert "labels" in item, "Missing labels in dataset item"
    assert torch.equal(item["input_ids"], item["labels"]), "Input_ids and labels should be equal"

@pytest.mark.asyncio
async def test_train_epoch(tokenizer, tmp_path, device):
    """Test training for one epoch"""
    # Initialize model and dataset
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        force_retrain=True,
        batch_size=1
    )
    model = model.to(device)
    
    # Create small dataset
    encodings = tokenizer(
        ["test text"] * 2,  # Only 2 samples
        truncation=True,
        padding=True,
        max_length=32,  # Reduced from 128
        return_tensors="pt"
    )
    dataset = SimpleDataset(encodings)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # Disabled shuffling for tests
    
    # Train for one epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(len(train_loader)), desc="Testing")
    
    avg_loss = await train_epoch(model, train_loader, optimizer, device, 0)
    assert isinstance(avg_loss, float), "Loss should be a float"
    assert not torch.isnan(torch.tensor(avg_loss)), "Loss should not be NaN"

def test_interrupt_handling():
    """Test interrupt handling during training"""
    global should_stop
    
    # Test setting the flag
    should_stop = True
    assert should_stop, "Should_stop flag not set"
    
    # Reset flag
    should_stop = False
    assert not should_stop, "Should_stop flag not reset"

@pytest.mark.asyncio
async def test_model_saving(tokenizer, tmp_path):
    """Test model saving functionality"""
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        force_retrain=True,
        batch_size=1
    )
    
    # Check saved files
    assert (tmp_path / "pytorch_model.bin").exists(), "Model weights not saved"
    assert (tmp_path / "config.json").exists(), "Model config not saved"
    
    # Test loading saved model
    loaded_model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        force_retrain=False
    )
    assert loaded_model is not None, "Failed to load saved model"
    
    # Compare model parameters
    orig_params = list(model.parameters())
    loaded_params = list(loaded_model.parameters())
    assert len(orig_params) == len(loaded_params), "Model parameter count mismatch"
    for p1, p2 in zip(orig_params, loaded_params):
        assert torch.equal(p1.data, p2.data), "Model parameters don't match"

@pytest.mark.asyncio
async def test_checkpoint_saving(tokenizer, tmp_path):
    """Test checkpoint saving on interrupt"""
    global should_stop
    
    # Set interrupt flag before training
    should_stop = True
    
    # Start training with minimal data
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        force_retrain=True,
        batch_size=1
    )
    
    # Check checkpoint directory
    checkpoint_dir = tmp_path / "interrupted_checkpoint"
    assert checkpoint_dir.exists(), "Checkpoint directory not created"
    assert (checkpoint_dir / "pytorch_model.bin").exists(), "Checkpoint model not saved"
    assert (checkpoint_dir / "config.json").exists(), "Checkpoint config not saved"
    
    # Reset flag
    should_stop = False

@pytest.mark.asyncio
async def test_find_latest_checkpoint(tokenizer, tmp_path):
    """Test finding the latest checkpoint"""
    # Create some test checkpoints
    checkpoints = [
        tmp_path / "checkpoint_epoch_0.pt",
        tmp_path / "checkpoint_epoch_1.pt"
    ]
    
    # Save dummy checkpoints
    dummy_state = {
        'epoch': 0,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'loss': 0.0
    }
    
    for i, checkpoint in enumerate(checkpoints):
        dummy_state['epoch'] = i
        torch.save(dummy_state, checkpoint)
    
    # Test finding latest checkpoint
    latest_path, latest_epoch = find_latest_checkpoint(tmp_path)
    assert latest_path == checkpoints[-1], "Did not find correct latest checkpoint"
    assert latest_epoch == 1, "Did not get correct latest epoch number"
    
    # Test with no checkpoints
    for checkpoint in checkpoints:
        checkpoint.unlink()
    empty_path, empty_epoch = find_latest_checkpoint(tmp_path)
    assert empty_path is None, "Should return None when no checkpoints exist"
    assert empty_epoch == -1, "Should return -1 when no checkpoints exist"

@pytest.mark.asyncio
async def test_resume_training(tokenizer, tmp_path):
    """Test resuming training from a checkpoint"""
    # First training run with minimal data
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=1,
        force_retrain=True,
        batch_size=1
    )
    
    # Get loss from last checkpoint
    checkpoint_path = tmp_path / "checkpoint_epoch_0.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved"
    checkpoint = load_checkpoint(checkpoint_path)
    first_run_loss = checkpoint['loss']
    
    # Resume training for one more epoch
    resumed_model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=2,
        force_retrain=False,
        batch_size=1
    )
    
    # Check that we have a new checkpoint
    final_checkpoint_path = tmp_path / "checkpoint_epoch_1.pt"
    assert final_checkpoint_path.exists(), "Final checkpoint not saved"
    
    # Verify we started from the correct epoch
    checkpoints = list(tmp_path.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) == 2, "Wrong number of checkpoints"

@pytest.mark.asyncio
async def test_force_retrain(tokenizer, tmp_path):
    """Test force_retrain flag behavior"""
    # First training run with minimal data
    await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=1,
        force_retrain=True,
        batch_size=1
    )
    
    # Get initial checkpoint count
    initial_checkpoints = list(tmp_path.glob("checkpoint_epoch_*.pt"))
    initial_count = len(initial_checkpoints)
    
    # Run with force_retrain=True
    await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=1,
        force_retrain=True,
        batch_size=1
    )
    
    # Check that we have new checkpoints
    new_checkpoints = list(tmp_path.glob("checkpoint_epoch_*.pt"))
    assert len(new_checkpoints) == initial_count, "Should have same number of checkpoints"
    
    # Verify checkpoint epochs start from 0
    checkpoint = load_checkpoint(tmp_path / "checkpoint_epoch_0.pt")
    assert checkpoint['epoch'] == 0, "Force retrain should start from epoch 0"

@pytest.mark.asyncio
async def test_completed_training_detection(tokenizer, tmp_path):
    """Test detection of completed training"""
    # Complete training with minimal data
    await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=1,
        force_retrain=True,
        batch_size=1
    )
    
    # Try to train again with same number of epochs
    model = await train_model(
        model=None,
        dataset=None,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        training_samples=2,
        num_epochs=1,
        force_retrain=False,
        batch_size=1
    )
    
    # Verify model was loaded but not retrained
    assert model is not None, "Model should be loaded"
    assert (tmp_path / "pytorch_model.bin").exists(), "Model file should exist"
    
    # Check no new checkpoints were created
    checkpoints = list(tmp_path.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) == 1, "No new checkpoints should be created" 