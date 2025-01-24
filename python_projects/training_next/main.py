import asyncio
import logging
import signal
import os
from pathlib import Path
from training_next.dataset_utils import load_and_prepare_dataset, monitor_cpu_usage
from training_next.tokenizer_trainer import train_tokenizer
from training_next.model_trainer import train_model, GPT2LMHeadModel, GPT2Config

# Get the directory where the script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Constants
FORCE_RETRAIN = False  # Set to True to ignore existing tokenizer/model and retrain
TOKENIZER_SAMPLES = 20000  # Number of examples to use for tokenizer training
TRAINING_SAMPLES = 1000  # Number of examples to use for model training
VOCAB_SIZE = 8192  # Size of the tokenizer vocabulary
MODEL_OUTPUT_DIR = SCRIPT_DIR / "tiny_gpt"  # Directory to save the trained model and tokenizer
USE_GPU = False  # Using GPU for faster training
MAX_CPU_CORES = 2  # Maximum number of CPU cores to use

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
    if should_stop:  # If already stopping, exit immediately
        log.info("Forced exit.")
        os._exit(1)
    log.info("Interrupt signal received (SIGINT). Stopping gracefully...")
    should_stop = True

async def main():
    """Main function to control the training process."""
    global should_stop
    should_stop = False  # Reset the flag at the start
    
    try:
        # Register the interrupt handler
        signal.signal(signal.SIGINT, handle_interrupt)
        
        # Change to script directory
        os.chdir(SCRIPT_DIR)
        
        if not MODEL_OUTPUT_DIR.exists():
            MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initial CPU usage
        log.info("Initial CPU usage:")
        monitor_cpu_usage()
        
        dataset = await load_and_prepare_dataset(tokenizer_samples=TOKENIZER_SAMPLES)
        if should_stop:
            return
        
        tokenizer = await train_tokenizer(dataset, VOCAB_SIZE, output_dir=MODEL_OUTPUT_DIR, force_retrain=FORCE_RETRAIN)
        if should_stop or tokenizer is None:
            return
            
        # Initialize model
        if FORCE_RETRAIN or not (MODEL_OUTPUT_DIR / "pytorch_model.bin").exists():
            config = GPT2Config(
                vocab_size=VOCAB_SIZE,
                n_positions=128,
                n_ctx=128,
                n_embd=256,
                n_layer=6,
                n_head=8,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel.from_pretrained(str(MODEL_OUTPUT_DIR))
        
        # Train model
        model = await train_model(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            output_dir=MODEL_OUTPUT_DIR,
            batch_size=32,
            num_epochs=3,
            learning_rate=3e-4,
            training_samples=TRAINING_SAMPLES,
            max_cpu_cores=MAX_CPU_CORES,
            force_retrain=FORCE_RETRAIN
        )
        
        if should_stop:
            log.info("Training interrupted by user.")
            return
            
        if not should_stop:
            log.info("Training complete!")
            
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
        should_stop = True
    except Exception as e:
        log.error(f"Error during training: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
    finally:
        if should_stop:
            log.info("Cleaning up after interruption...")
            
        # Final CPU usage
        log.info("Final CPU usage:")
        monitor_cpu_usage()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Main interrupt is handled inside main()