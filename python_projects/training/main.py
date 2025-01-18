from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    GPT2Config,
    PreTrainedTokenizerFast
)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import torch
import logging
import asyncio
import signal
import json
import os
import time
from pathlib import Path
from functools import partial

# Constants
FORCE_RETRAIN = True    # Set to True to ignore existing tokenizer/model and retrain
TOKENIZER_SAMPLES = 500000  # Number of examples to use for tokenizer training
TRAINING_SAMPLES = 2000    # Number of examples to use for model training
VOCAB_SIZE = 8192         # Size of the tokenizer vocabulary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("rich")

# Global flag for graceful shutdown
should_stop = False

def handle_interrupt(signum, frame):
    """Handle interrupt signal gracefully"""
    global should_stop
    log.info("Received interrupt signal. Stopping gracefully...")
    should_stop = True

def format_time(seconds):
    """Format seconds into human readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:.0f}h {minutes:.0f}m"
    elif minutes > 0:
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        return f"{seconds:.0f}s"

async def train_tokenizer(dataset, vocab_size=8192, output_dir=None):
    """Train a new BPE tokenizer on the dataset or load if exists."""
    tokenizer_path = output_dir / "tokenizer.json" if output_dir else None
    state_path = output_dir / "tokenizer_state.json" if output_dir else None
    
    if not FORCE_RETRAIN and tokenizer_path and tokenizer_path.exists():
        log.info("Loading existing tokenizer...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
        
    log.info(f"Starting tokenizer training with vocab_size={vocab_size}")
    log.info(f"Dataset size: {len(dataset)} examples")
    log.info(f"First example text: {dataset[0]['text'][:100]}...")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Track progress and metrics
    processed_count = 0
    batch_size = 1000
    total_size = len(dataset)
    start_time = time.time()
    last_log_time = start_time
    training_completed = False
    total_tokens = 0
    total_chars = 0
    
    # Load previous state if exists
    if state_path and state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
                processed_count = state.get('processed_count', 0)
                if processed_count > 0:
                    log.info(f"Resuming tokenizer training from {processed_count}/{total_size} examples")
        except json.JSONDecodeError:
            log.info("Invalid state file, starting from beginning")
            processed_count = 0
    
    def get_training_corpus():
        """Regular generator for tokenizer training"""
        nonlocal processed_count, last_log_time, training_completed, total_tokens, total_chars
        log.info("Starting to yield training corpus...")
        
        # Create a single temp tokenizer for metrics
        temp_tokenizer = Tokenizer(models.BPE())
        temp_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        for i in range(processed_count, len(dataset), batch_size):
            if should_stop:
                log.info("Training corpus generation interrupted")
                # Save state before stopping
                if state_path:
                    try:
                        with open(state_path, 'w') as f:
                            json.dump({'processed_count': i}, f)
                        log.info(f"Saved tokenizer state at {i}/{total_size} examples")
                    except Exception as e:
                        log.error(f"Failed to save state: {e}")
                return
                
            batch = dataset[i:i + batch_size]["text"]
            batch_size_actual = len(batch)
            processed_count = i + batch_size_actual
            
            # Update progress and metrics every 10 seconds
            current_time = time.time()
            if current_time - last_log_time > 10:
                elapsed = current_time - start_time
                progress = i / total_size
                if progress > 0:
                    remaining = elapsed / progress - elapsed
                    speed = i / elapsed
                    
                    # Calculate metrics on a sample of the batch
                    sample_size = min(10, len(batch))  # Use at most 10 examples for metrics
                    sample_chars = 0
                    sample_tokens = 0
                    for text in batch[:sample_size]:
                        sample_chars += len(text)
                        tokens = temp_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                        sample_tokens += len(tokens)
                    
                    # Calculate metrics from sample
                    avg_tokens_per_char = sample_tokens / sample_chars if sample_chars > 0 else 0
                    compression = 1 / avg_tokens_per_char if avg_tokens_per_char > 0 else 0
                    
                    log.info(f"Progress: {i}/{total_size} ({progress:.1%}) - {speed:.0f} examples/s - ETA: {format_time(remaining)}")
                    log.info(f"Metrics: Compression ratio: {compression:.2f}x (avg {avg_tokens_per_char:.2f} tokens/char)")
                    log.info(f"Vocab size so far: {len(tokenizer.get_vocab())}")
                last_log_time = current_time
            
            yield batch
            
            # Check if we've processed everything
            remaining = total_size - processed_count
            if remaining <= 0:
                training_completed = True
                log.info(f"Processed all {total_size} examples!")
                log.info(f"Final processed_count: {processed_count}")
                
                # Calculate final metrics on a larger sample
                sample_size = min(100, len(dataset))  # Use 100 examples for final metrics
                sample_chars = 0
                sample_tokens = 0
                for text in dataset[:sample_size]["text"]:
                    sample_chars += len(text)
                    tokens = temp_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                    sample_tokens += len(tokens)
                
                # Calculate final metrics
                avg_tokens_per_char = sample_tokens / sample_chars if sample_chars > 0 else 0
                compression = 1 / avg_tokens_per_char if avg_tokens_per_char > 0 else 0
                log.info(f"Final metrics: Compression ratio: {compression:.2f}x (avg {avg_tokens_per_char:.2f} tokens/char)")
                log.info(f"Final vocab size: {len(tokenizer.get_vocab())}")
                return
    
    try:
        # Run tokenizer training in thread pool to not block
        log.info("Starting tokenizer training...")
        
        async def train():
            nonlocal last_log_time
            while not should_stop:
                try:
                    # Reset log time on each attempt
                    last_log_time = time.time()
                    # Train in small chunks to allow interruption
                    corpus = get_training_corpus()
                    tokenizer.train_from_iterator(corpus, trainer=trainer)
                    break
                except Exception as e:
                    if should_stop:
                        log.info("Tokenizer training interrupted")
                        return None
                    if "mutably borrowed" not in str(e):
                        raise e
                    # If we hit a borrow error, wait a bit and retry
                    await asyncio.sleep(0.1)
                    continue
            
            return tokenizer if not should_stop else None
            
        tokenizer = await train()
        
        if tokenizer is None:
            log.info("Tokenizer training was interrupted")
            return None
        
        log.info(f"Tokenizer after training: {tokenizer}")
        log.info("Tokenizer training completed")
        
        # Force training_completed if we processed everything
        if processed_count >= total_size:
            training_completed = True
            log.info("Forcing training completion flag - all examples processed")
        
        # Save if training completed successfully
        if training_completed:
            log.info("Tokenizer training completed successfully!")
            if tokenizer_path:
                log.info(f"Saving tokenizer to {tokenizer_path}...")
                tokenizer.save(str(tokenizer_path))
                log.info("Tokenizer saved successfully")
            # Clear state file if training completed
            if state_path and state_path.exists():
                try:
                    os.remove(state_path)
                    log.info("Cleared tokenizer state file")
                except Exception as e:
                    log.error(f"Failed to remove state file: {e}")
            return tokenizer
        elif should_stop:
            log.info("Tokenizer training was interrupted.")
            return None
                
        log.info("Training incomplete")
        return None
        
    except Exception as e:
        log.error(f"Error during tokenizer training: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_model_config(vocab_size=8192):
    """Get configuration for a small GPT model."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        max_position_embeddings=128,  # Same as block_size
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )

class InterruptibleTrainer(Trainer):
    """Custom trainer that can be interrupted gracefully"""
    def training_step(self, *args, **kwargs):
        if should_stop:
            raise KeyboardInterrupt
        return super().training_step(*args, **kwargs)

async def async_main():
    try:
        # Get total dataset size first
        log.info("Checking total dataset size...")
        loop = asyncio.get_event_loop()
        full_dataset = await loop.run_in_executor(
            None,
            lambda: load_dataset("roneneldan/TinyStories", split="train")
        )
        log.info(f"Total samples in TinyStories dataset: {len(full_dataset):,}")
        
        # Create output directory
        output_dir = Path("./tiny_gpt")
        output_dir.mkdir(exist_ok=True)
        
        # Check for existing HF tokenizer
        hf_tokenizer_path = output_dir / "tokenizer_config.json"
        if not FORCE_RETRAIN and hf_tokenizer_path.exists():
            log.info("Loading existing HuggingFace tokenizer...")
            hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(output_dir))
        else:
            if FORCE_RETRAIN:
                log.info("Force retrain enabled - training new tokenizer...")
            log.info(f"Loading {TOKENIZER_SAMPLES} samples for tokenizer training...")
            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(
                None,
                lambda: load_dataset("roneneldan/TinyStories", split=f"train[:{TOKENIZER_SAMPLES}]")
            )
            
            log.info(f"Using {len(dataset)} examples for tokenizer training")
            
            if should_stop:
                return
            
            # Train or load tokenizer
            tokenizer = await train_tokenizer(dataset, vocab_size=VOCAB_SIZE, output_dir=output_dir)
            
            if should_stop or tokenizer is None:
                log.info("Tokenizer training was interrupted or failed. Exiting...")
                return
                
            # Convert to Hugging Face tokenizer
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token="<|endoftext|>",
                eos_token="<|endoftext|>",
                pad_token="<|pad|>"
            )
            hf_tokenizer.save_pretrained(output_dir)
        
        log.info("Initializing model...")
        config = get_model_config(vocab_size=hf_tokenizer.vocab_size)
        
        # Check for existing model checkpoint
        if not FORCE_RETRAIN and (output_dir / "pytorch_model.bin").exists():
            log.info("Loading existing model checkpoint...")
            model = AutoModelForCausalLM.from_pretrained(str(output_dir))
        else:
            if FORCE_RETRAIN:
                log.info("Force retrain enabled - initializing new model...")
            model = AutoModelForCausalLM.from_config(config)
        
        if should_stop:
            return
            
        # Only tokenize dataset if not resuming from checkpoint or force retrain
        if FORCE_RETRAIN or not (output_dir / "trainer_state.json").exists():
            log.info(f"Loading and tokenizing dataset ({TRAINING_SAMPLES} samples) for model training...")
            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(
                None,
                lambda: load_dataset("roneneldan/TinyStories", split=f"train[:{TRAINING_SAMPLES}]")
            )
            
            def tokenize_function(examples):
                # Tokenize the texts
                tokenized = hf_tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=128,
                    padding="max_length"
                )
                
                # Create labels by shifting input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()  # Next token prediction
                
                return tokenized
            
            tokenized_dataset = await loop.run_in_executor(
                None,
                lambda: dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=dataset.column_names
                )
            )
        else:
            log.info("Resuming from existing checkpoint...")
            tokenized_dataset = None  # Trainer will load from cache
        
        if should_stop:
            return
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=32,
            save_steps=10,  # Save more frequently for testing
            save_total_limit=2,
            logging_steps=1,  # Log every step for testing
            save_strategy="steps",
            resume_from_checkpoint=True,
            no_cuda=True,  # Force CPU
            log_level="info"
        )
        
        log.info("Starting/Resuming training...")
        trainer = InterruptibleTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        try:
            await loop.run_in_executor(None, trainer.train)
        except KeyboardInterrupt:
            log.info("Training interrupted. Saving checkpoint...")
            trainer.save_model()
            trainer.save_state()
            return
            
        if not should_stop:
            log.info("Saving model and tokenizer...")
            model.save_pretrained(output_dir)
            hf_tokenizer.save_pretrained(output_dir)
            log.info("Training complete!")
    
    except Exception as e:
        log.exception(f"Error during training: {e}")
    finally:
        if should_stop:
            log.info("Training stopped gracefully.")

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    
    # Run the async main
    asyncio.run(async_main())

if __name__ == "__main__":
    main()