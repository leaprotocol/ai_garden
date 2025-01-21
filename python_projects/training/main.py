import asyncio
import logging
import signal
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast

# Constants
FORCE_RETRAIN = True  # Set to True to ignore existing tokenizer/model and retrain
TOKENIZER_SAMPLES = 50000  # Number of examples to use for tokenizer training
TRAINING_SAMPLES = 2000  # Number of examples to use for model training
VOCAB_SIZE = 8192  # Size of the tokenizer vocabulary
MODEL_OUTPUT_DIR = "tiny_gpt"  # Directory to save the trained model and tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# Global flag to signal training tasks to stop
should_stop = False

def handle_interrupt(signum, frame):
    """Handle interrupt signals gracefully."""
    global should_stop
    log.info("Interrupt signal received (SIGINT).")
    should_stop = True

async def load_and_prepare_dataset(dataset_name="roneneldan/TinyStories", split="train", tokenizer_samples=None):
    """Load the dataset and sample for tokenizer training."""
    log.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    total_samples = len(dataset)
    log.info(f"Total samples in {split} dataset: {total_samples}")

    if tokenizer_samples:
        if tokenizer_samples > total_samples:
            log.warning(f"Requested {tokenizer_samples} samples, but dataset only has {total_samples}. Using all samples.")
            tokenizer_samples = total_samples
        else:
            log.info(f"Using {tokenizer_samples} samples for tokenizer training")
            dataset = dataset.select(range(tokenizer_samples))

    return dataset

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
        
        for i in range(processed_count, total_size, batch_size):
            batch = dataset[i:i + batch_size]
            
            # Check for stop signal between batches
            if should_stop:
                log.info("Stop signal received, pausing tokenizer training...")
                return
            
            texts = [example["text"] for example in batch]
            
            # Use temp tokenizer to get metrics for this batch
            encoded_batch = temp_tokenizer.encode_batch(texts)
            batch_tokens = sum(len(encoding.ids) for encoding in encoded_batch)
            batch_chars = sum(len(text) for text in texts)
            total_tokens += batch_tokens
            total_chars += batch_chars
            
            yield texts
            processed_count += len(batch)

            # Calculate and log metrics
            current_time = time.time()
            elapsed_time = current_time - start_time
            if current_time - last_log_time >= 5:
                avg_tokens_per_example = total_tokens / processed_count if processed_count else 0
                avg_chars_per_example = total_chars / processed_count if processed_count else 0
                examples_per_second = processed_count / elapsed_time if elapsed_time else 0
                
                metrics = {
                    "processed_examples": processed_count,
                    "elapsed_time": int(elapsed_time),
                    "examples_per_second": int(examples_per_second),
                    "avg_tokens_per_example": round(avg_tokens_per_example, 2),
                    "avg_chars_per_example": round(avg_chars_per_example, 2),
                }
                log.info(f"Tokenizer training metrics: {metrics}")
                last_log_time = current_time
    
    try:
        # Wrap the get_training_corpus generator to handle exceptions
        def safe_training_corpus():
            try:
                yield from get_training_corpus()
            except Exception as e:
                log.error(f"Error in get_training_corpus: {e}")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()}")
                return
        
        # Train in batches
        tokenizer.train_from_iterator(
            safe_training_corpus(),
            trainer=trainer,
            length=total_size,
        )
        
        log.info(f"Tokenizer after training: {tokenizer}")
        
        # Save tokenizer state if not completed
        if not training_completed and state_path:
            state = {
                'processed_count': processed_count,
            }
            with open(state_path, 'w') as f:
                json.dump(state, f)
            log.info(f"Saved tokenizer state to {state_path}")
        
        log.info(f"Tokenizer training metrics: {metrics}")
        
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
                
                # Verify tokenizer integrity
                log.info("Verifying tokenizer integrity...")
                loaded_tokenizer = Tokenizer.from_file(str(tokenizer_path))
                if tokenizer.to_str() == loaded_tokenizer.to_str():
                    log.info("Tokenizer integrity verified!")
                else:
                    log.error("Tokenizer integrity check failed!")

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

async def train_model(dataset, tokenizer, output_dir, training_samples=None):
    """Train a GPT model on the dataset."""
    log.info("Initializing model...")
    
    if not FORCE_RETRAIN and (output_dir / "pytorch_model.bin").exists():
        log.info("Loading existing model...")
        model = AutoModelForCausalLM.from_pretrained(str(output_dir))
    else:
        log.info("Force retrain enabled - initializing new model...")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tokenizer.pad_token = "<|pad|>"  # Set pad token here
        tokenizer.unk_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.eos_token = "<|endoftext|>"
        model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.pad_token_id)
        
    if training_samples:
        log.info(f"Loading and tokenizing dataset ({training_samples} samples) for model training...")
        dataset = dataset.select(range(training_samples))
    else:
        log.info("Loading and tokenizing entire dataset for model training...")
    
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="tensorboard",
        learning_rate=5e-5,  # Add a learning rate
        adam_epsilon=1e-8,  # Set epsilon for Adam optimizer
        no_cuda=True,
        gradient_accumulation_steps=1,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        eval_dataset=tokenized_dataset,
    )

    log.info("Starting/Resuming training...")
    try:
        trainer.train()
    except Exception as e:
        log.error(f"Error during model training: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return None

    return model

async def main():
    """Main function to control the training process."""
    global should_stop
    
    # Register the interrupt handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    output_dir = Path(MODEL_OUTPUT_DIR)
    if not output_dir.exists():
        os.makedirs(output_dir)
    
    dataset = await load_and_prepare_dataset(tokenizer_samples=TOKENIZER_SAMPLES)
    if should_stop:
        return
    
    tokenizer = await train_tokenizer(dataset, VOCAB_SIZE, output_dir=output_dir)
    if should_stop or tokenizer is None:
        return
    
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<|pad|>"
    hf_tokenizer.unk_token = "<|endoftext|>"
    hf_tokenizer.bos_token = "<|endoftext|>"
    hf_tokenizer.eos_token = "<|endoftext|>"
    
    model = await train_model(dataset, hf_tokenizer, output_dir, training_samples=TRAINING_SAMPLES)
    if should_stop:
        return
        
    if not should_stop:
        log.info("Saving model and tokenizer...")
        model.save_pretrained(output_dir)
        hf_tokenizer.save_pretrained(output_dir)
        log.info(f"Model and tokenizer saved to {output_dir}")
        log.info("Training complete!")

if __name__ == "__main__":
    asyncio.run(main())