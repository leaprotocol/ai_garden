import torch
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import AsyncIterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("smol_hello")
console = Console()

def load_model(model_name: str = "HuggingFaceTB/SmolLM2-360M"):
    """Load model and tokenizer with optimizations for GTX 1060."""
    try:
        log.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with FP16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Move to GPU and eval mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        if device == "cuda":
            log.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
        return model, tokenizer
        
    except Exception as e:
        log.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def generate_text(model, tokenizer, prompt: str, max_length: int = 50):
    """Generate text from prompt."""
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Clear CUDA cache before generation
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # Generate with memory-efficient settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
            
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        log.error(f"Error during generation: {str(e)}", exc_info=True)
        raise

async def generate_text_stream(model, tokenizer, prompt: str, max_length: int = 50) -> AsyncIterator[str]:
    """Generate text from prompt with streaming output."""
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Clear CUDA cache before generation
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Generate with streaming
        generated_ids = []
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + 1,  # Only generate one token at a time
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Get the last generated token
                next_token = outputs[0][-1].unsqueeze(0)
                
                # Decode the token
                token_text = tokenizer.decode(next_token, skip_special_tokens=True)
                
                # If we got an empty string or end token, stop
                if not token_text or token_text == tokenizer.eos_token:
                    break
                    
                # Yield the token
                yield token_text
                
                # Update input_ids for next iteration
                inputs.input_ids = outputs
                
    except Exception as e:
        log.error(f"Error during streaming generation: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Test prompts
        prompts = [
            "Hello, my name is",
            "The meaning of life is",
            "Once upon a time"
        ]
        
        # Generate and display results
        for prompt in prompts:
            console.rule(f"[bold cyan]Generating from: {prompt}")
            response = generate_text(model, tokenizer, prompt)
            console.print(f"[green]Response:[/green] {response}\n")
            
    except KeyboardInterrupt:
        log.info("Gracefully shutting down...")
    except Exception as e:
        log.error("An error occurred", exc_info=True)
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 