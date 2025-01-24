import asyncio
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedTokenizerFast, 
    GPT2Config, 
    GPT2LMHeadModel
)
from tqdm.auto import tqdm
from training_next.dataset_utils import (
    load_and_prepare_dataset, 
    get_optimal_num_workers, 
    monitor_cpu_usage,
    set_cpu_affinity,
    worker_init_fn
)
import torch.multiprocessing as mp
import os
from typing import Optional, List, Dict, Tuple
import functools
import glob

# Set tokenizer parallelism explicitly
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure PyTorch thread settings
def configure_torch_threads(num_cores: int):
    """Configure PyTorch to use specific number of threads"""
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# Global flag to signal training tasks to stop
should_stop = False

class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings["input_ids"])
        
    def __getitem__(self, idx):
        # Use clone().detach() directly on tensor values
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

def create_dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool = True,
    cores: Optional[List[int]] = None
) -> DataLoader:
    """Create a DataLoader with CPU affinity and optimal multiprocessing settings"""
    num_workers = len(cores) if cores else get_optimal_num_workers()
    
    # Create worker init function with core assignments
    worker_init = functools.partial(worker_init_fn, cores=cores) if cores else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        worker_init_fn=worker_init
    )

async def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    global should_stop
    model.train()
    total_loss = 0
    
    # Use tqdm with dynamic ncols for better display
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    
    try:
        for batch_idx, batch in enumerate(progress):
            if should_stop:
                log.info("Stopping training due to interrupt signal")
                raise KeyboardInterrupt  # Propagate interrupt to train_model
                
            # Move batch to device efficiently
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Monitor CPU usage every 10 batches
            if batch_idx % 10 == 0:
                monitor_cpu_usage()
                # Allow other async operations to run
                await asyncio.sleep(0)
                
    except KeyboardInterrupt:
        log.info("Received interrupt signal during training")
        should_stop = True
        raise  # Re-raise to propagate to train_model
        
    return total_loss / len(dataloader)

def find_latest_checkpoint(output_dir: Path) -> Tuple[Optional[Path], int]:
    """Find the latest checkpoint file and its epoch number"""
    checkpoint_files = glob.glob(str(output_dir / "checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None, -1
        
    # Extract epoch numbers and find the latest
    epoch_numbers = [int(f.split("_")[-1].replace(".pt", "")) for f in checkpoint_files]
    latest_idx = max(range(len(epoch_numbers)), key=epoch_numbers.__getitem__)
    return Path(checkpoint_files[latest_idx]), epoch_numbers[latest_idx]

def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a checkpoint file"""
    log.info(f"Loading checkpoint from {checkpoint_path}")
    return torch.load(checkpoint_path)

async def train_model(
    model,
    dataset,
    tokenizer,
    output_dir: Path,
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    training_samples: int = 2000,
    max_cpu_cores: Optional[int] = None,
    force_retrain: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the model with efficient data loading and processing
    
    Args:
        output_dir: Directory to save checkpoints and final model
        force_retrain: If True, start training from scratch
        max_cpu_cores: Maximum number of CPU cores to use. If None, uses system limit.
    """
    global should_stop
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model is already fully trained
    model_path = output_dir / "pytorch_model.bin"
    if not force_retrain and model_path.exists():
        log.info("Loading existing model...")
        if model is None:
            config = GPT2Config.from_pretrained(str(output_dir))
            model = GPT2LMHeadModel(config)
        model.load_state_dict(torch.load(model_path))
        log.info("Model loaded successfully!")
        return model
    
    # Configure thread and CPU usage
    if max_cpu_cores:
        configure_torch_threads(max_cpu_cores)
        cores = list(range(max_cpu_cores))
        set_cpu_affinity(cores)
        log.info(f"Main process restricted to cores: {cores}")
        log.info(f"PyTorch threads limited to: {max_cpu_cores}")
    
    if dataset is None:
        log.info("No dataset provided, loading default dataset...")
        dataset = await load_and_prepare_dataset(streaming=True)
    
    # Convert streaming dataset to SimpleDataset for training
    log.info("Processing dataset for training...")
    texts = [item["text"] for item in dataset.take(training_samples)]
    log.info(f"Processing {len(texts)} texts for training...")
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    train_dataset = SimpleDataset(encodings)
    
    # Initialize model if None
    if model is None:
        log.info("Initializing new model...")
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=128,
            n_ctx=128,
            n_embd=256,
            n_layer=6,
            n_head=8,
            pad_token_id=tokenizer.pad_token_id
        )
        model = GPT2LMHeadModel(config)
        log.info("Model initialized successfully")
    
    # Set up multiprocessing for PyTorch
    if device == "cuda":
        # Use all available GPUs
        model = nn.DataParallel(model)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Find latest checkpoint if not forcing retrain
    start_epoch = 0
    if not force_retrain:
        checkpoint_path, last_epoch = find_latest_checkpoint(output_dir)
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = last_epoch + 1
            log.info(f"Resuming training from epoch {start_epoch}")
            
            if start_epoch >= num_epochs:
                log.info("Training already completed (all epochs done). Use force_retrain=True to retrain.")
                return model
    
    # Create efficient dataloader with core assignments
    cores = list(range(max_cpu_cores)) if max_cpu_cores else None
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        cores=cores
    )
    
    log.info(f"Starting training on device: {device}")
    log.info(f"Number of workers: {len(cores) if cores else get_optimal_num_workers()}")
    if cores:
        log.info(f"CPU cores restricted to: {cores}")
    
    # Initial CPU usage
    monitor_cpu_usage()
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if should_stop:
                log.info("Training stopped due to interrupt signal")
                # Save interrupted checkpoint
                interrupted_dir = output_dir / "interrupted_checkpoint"
                interrupted_dir.mkdir(exist_ok=True)
                # Save model state and config
                torch.save(model.state_dict(), interrupted_dir / "pytorch_model.bin")
                model.config.save_pretrained(interrupted_dir)
                log.info(f"Saved interrupted model to {interrupted_dir}")
                return model
                
            avg_loss = await train_epoch(model, train_dataloader, optimizer, device, epoch)
            log.info(f"Epoch {epoch} completed with average loss: {avg_loss:.4f}")
            
            # Monitor CPU usage after each epoch
            monitor_cpu_usage()
            
            # Save checkpoint after each epoch
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            log.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save final model
            if epoch == num_epochs - 1:
                # Save model state
                torch.save(model.state_dict(), model_path)
                # Save config
                model.config.save_pretrained(output_dir)
                log.info(f"Saved model to {output_dir}")
            
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        should_stop = True
        # Save interrupted checkpoint
        interrupted_dir = output_dir / "interrupted_checkpoint"
        interrupted_dir.mkdir(exist_ok=True)
        # Save model state and config
        torch.save(model.state_dict(), interrupted_dir / "pytorch_model.bin")
        model.config.save_pretrained(interrupted_dir)
        log.info(f"Saved interrupted model to {interrupted_dir}")
        
    return model