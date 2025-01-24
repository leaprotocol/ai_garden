import logging
import json
import os
import time
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# Global flag to signal training tasks to stop
should_stop = False

def create_tokenizer(tokenizer_path: Path) -> Optional[PreTrainedTokenizerFast]:
    """Create a tokenizer from a saved file."""
    try:
        if not tokenizer_path.exists():
            log.error(f"Tokenizer file not found at {tokenizer_path}")
            return None
            
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        log.info("Tokenizer loaded successfully")
        return tokenizer
        
    except Exception as e:
        log.error(f"Error loading tokenizer: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return None

async def train_tokenizer(dataset, vocab_size: int = 8192, output_dir: Optional[Path] = None, force_retrain: bool = False, max_samples: int = 10000) -> PreTrainedTokenizerFast:
    """Train a BPE tokenizer on the dataset."""
    tokenizer_path = output_dir / "tokenizer.json" if output_dir else None
    state_path = output_dir / "tokenizer_state.json" if output_dir else None
    
    # Check for existing tokenizer
    if not force_retrain and tokenizer_path and tokenizer_path.exists():
        log.info("Loading existing tokenizer...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(output_dir))
        log.info("Tokenizer loaded successfully!")
        return tokenizer
    
    log.info(f"Starting tokenizer training with vocab_size={vocab_size}")
    log.info(f"Will process up to {max_samples} examples")
    
    # Create base tokenizer
    base_tokenizer = Tokenizer(models.BPE())
    base_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Track progress and metrics
    processed_count = 0
    batch_size = 1000
    start_time = time.time()
    last_log_time = start_time
    training_completed = False
    total_tokens = 0
    total_chars = 0
    
    def get_training_corpus():
        """Regular generator for tokenizer training"""
        nonlocal processed_count, last_log_time, training_completed, total_tokens, total_chars
        log.info("Starting to yield training corpus...")

        # Create a single temp tokenizer for metrics
        temp_tokenizer = Tokenizer(models.BPE())
        temp_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Buffer for batch processing
        current_batch = []
        
        for example in dataset:
            if processed_count >= max_samples:
                break
                
            current_batch.append(example["text"])
            
            if len(current_batch) >= batch_size:
                # Check for stop signal
                if should_stop:
                    log.info("Stop signal received, pausing tokenizer training...")
                    return

                # Use temp tokenizer to get metrics for this batch
                encoded_batch = temp_tokenizer.encode_batch(current_batch)
                batch_tokens = sum(len(encoding.ids) for encoding in encoded_batch)
                batch_chars = sum(len(text) for text in current_batch)
                total_tokens += batch_tokens
                total_chars += batch_chars

                # Yield texts for training
                yield current_batch
                processed_count += len(current_batch)
                current_batch = []

                # Log metrics every 5 seconds
                current_time = time.time()
                elapsed_time = current_time - start_time
                if current_time - last_log_time >= 5:
                    metrics = {
                        "processed_examples": processed_count,
                        "elapsed_time": int(elapsed_time),
                        "examples_per_second": int(processed_count / elapsed_time) if elapsed_time else 0,
                        "avg_tokens_per_example": round(total_tokens / processed_count, 2) if processed_count else 0,
                        "avg_chars_per_example": round(total_chars / processed_count, 2) if processed_count else 0,
                    }
                    log.info(f"Tokenizer training metrics: {metrics}")
                    last_log_time = current_time
        
        # Process remaining batch
        if current_batch:
            yield current_batch
            processed_count += len(current_batch)
    
    try:
        # Train the tokenizer
        base_tokenizer.train_from_iterator(
            get_training_corpus(),
            trainer=trainer,
            length=max_samples  # Use max_samples as the expected length
        )
        
        # Calculate final metrics
        current_time = time.time()
        elapsed_time = current_time - start_time
        metrics = {
            "processed_examples": processed_count,
            "elapsed_time": int(elapsed_time),
            "examples_per_second": int(processed_count / elapsed_time) if elapsed_time else 0,
            "avg_tokens_per_example": round(total_tokens / processed_count, 2) if processed_count else 0,
            "avg_chars_per_example": round(total_chars / processed_count, 2) if processed_count else 0,
        }
        
        log.info(f"Tokenizer training metrics: {metrics}")
        
        # Force training_completed if we processed enough examples
        if processed_count >= max_samples:
            training_completed = True
            log.info("Forcing training completion flag - reached max_samples")
        
        # Save if training completed successfully
        if training_completed and output_dir:
            log.info("Tokenizer training completed successfully!")
            log.info(f"Saving tokenizer to {tokenizer_path}...")
            base_tokenizer.save(str(tokenizer_path))
            log.info("Tokenizer saved successfully")
            
            # Clear state file if training completed
            if state_path and state_path.exists():
                try:
                    os.remove(state_path)
                    log.info("Cleared tokenizer state file")
                except Exception as e:
                    log.error(f"Failed to remove state file: {e}")
        
        # Create HuggingFace tokenizer wrapper
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            pad_token="<|pad|>",
            padding_side="right",
            model_max_length=128
        )
        
        # Save the HuggingFace tokenizer
        if output_dir:
            tokenizer.save_pretrained(output_dir)
        
        return tokenizer
        
    except Exception as e:
        log.error(f"Error during tokenizer training: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return None