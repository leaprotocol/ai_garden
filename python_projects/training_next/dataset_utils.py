import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import logging
import asyncio
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def set_cpu_affinity(cores: List[int]):
    """Set CPU affinity for the current process"""
    try:
        p = psutil.Process()
        p.cpu_affinity(cores)
        log.info(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        log.warning(f"Could not set CPU affinity: {e}")

def get_optimal_num_workers(max_cores: Optional[int] = None) -> int:
    """
    Get optimal number of workers based on CPU cores and limits
    
    Args:
        max_cores: Maximum number of CPU cores to use. If None, uses system limit.
    """
    available_cores = mp.cpu_count()
    if max_cores is not None:
        available_cores = min(available_cores, max_cores)
    return min(available_cores, 8)  # Cap at 8 workers to prevent overhead

def get_cpu_usage() -> float:
    """Get current CPU usage percentage across all cores"""
    return psutil.cpu_percent(interval=0.1)

def monitor_cpu_usage():
    """Log current CPU usage"""
    cpu_percent = psutil.cpu_percent(percpu=True)
    avg_usage = sum(cpu_percent) / len(cpu_percent)
    per_core = " ".join([f"Core {i}: {usage:.1f}%" for i, usage in enumerate(cpu_percent)])
    log.info(f"CPU Usage - Average: {avg_usage:.1f}% | {per_core}")

def worker_init_fn(worker_id: int, cores: List[int]):
    """Initialize worker process with CPU affinity"""
    worker_core = cores[worker_id % len(cores)]
    set_cpu_affinity([worker_core])
    log.info(f"Worker {worker_id} assigned to core {worker_core}")

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask'])
        }

def prepare_dataset(file_path: str) -> List[str]:
    """Load and prepare text data from file using memory-efficient reading"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                texts.append(line.strip())
    return texts

def create_dataloaders(
    dataset: TextDataset,
    batch_size: int,
    train_split: float = 0.8,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with multiprocessing"""
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    # Use numpy for efficient shuffling
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    train_size = int(len(dataset) * train_split)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Enable multiprocessing in DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    log.info(f"Created dataloaders with {num_workers} workers")
    return train_loader, val_loader

async def load_and_prepare_dataset(
    dataset_name: str = "roneneldan/TinyStories", 
    split: str = "train", 
    tokenizer_samples: Optional[int] = None,
    seed: Optional[int] = None,
    streaming: bool = True  # Enable streaming by default
) -> load_dataset:
    """Load the dataset and sample for tokenizer training with efficient streaming."""
    log.info(f"Loading dataset {dataset_name}...")
    
    # Use streaming for memory efficiency
    dataset = load_dataset(
        dataset_name, 
        split=split,
        streaming=streaming
    )
    
    if tokenizer_samples:
        log.info(f"Using {tokenizer_samples} samples for tokenizer training")
        if seed is not None:
            # For streaming datasets, use built-in shuffle with reasonable buffer
            dataset = dataset.shuffle(seed=seed, buffer_size=1000)
        dataset = dataset.take(tokenizer_samples)
    
    return dataset