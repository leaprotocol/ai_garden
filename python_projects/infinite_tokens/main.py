"""Token generator using direct model forward pass for more control."""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console
from rich.progress import Progress
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/infinite_tokens.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class InfiniteTokenGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """Initialize the token generator with specified model"""
        self.console = Console()
        
        # Determine device (cuda, mps, or cpu)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        self.console.print(f"[bold green]Using device:[/bold green] {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.console.print(f"[bold]Loading model:[/bold] {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        logger.info("Model loaded successfully")
        
    def generate_infinite_tokens(self, initial_prompt="Once upon a time", max_new_tokens=20, temperature=0.7, stream_delay=0.1):
        """Generate and stream tokens infinitely starting with the given prompt"""
        prompt = initial_prompt
        total_tokens = 0
        past_key_values = None
        
        self.console.print(f"\n[bold blue]Starting infinite token generation[/bold blue]")
        self.console.print(f"[dim]Initial prompt:[/dim] {prompt}")
        logger.info(f"Starting infinite generation with prompt: {prompt}")
        
        # Print the initial prompt
        self.console.print(f"\n{prompt}", end="", highlight=False)
        
        try:
            # Tokenize initial prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            while True:  # Run indefinitely
                # Generate one token at a time
                with torch.no_grad():
                    # Forward pass through the model
                    outputs = self.model(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # Get logits and past key values for next iteration
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Get probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample next token
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Get top tokens and their probabilities for logging
                    topk_probs, topk_indices = torch.topk(probs, k=5)
                    top_tokens = [
                        (self.tokenizer.decode([idx.item()]), prob.item())
                        for idx, prob in zip(topk_indices[0], topk_probs[0])
                    ]
                    
                    # Decode the generated token
                    new_token = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    
                    # Print the token
                    self.console.print(new_token, end="", highlight=False)
                    sys.stdout.flush()
                    time.sleep(stream_delay)
                    
                    # Update input_ids for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                    
                    # Update total tokens count
                    total_tokens += 1
                    
                    # Log token info
                    logger.debug(f"Token {total_tokens}: '{new_token}' (p={top_tokens[0][1]:.4f})")
                    logger.debug(f"Top tokens: {top_tokens}")
                    
                    # Manage context length to prevent OOM
                    if input_ids.shape[1] > 2000:
                        # Keep the last 1000 tokens
                        input_ids = input_ids[:, -1000:]
                        attention_mask = attention_mask[:, -1000:]
                        # Reset past_key_values to force recomputation
                        past_key_values = None
                        logger.info("Trimmed context to last 1000 tokens")
                    
        except KeyboardInterrupt:
            self.console.print("\n\n[bold red]Generation stopped by user[/bold red]")
            logger.info(f"Generation stopped by user after {total_tokens} tokens")
            self.console.print(f"[bold]Total tokens generated:[/bold] {total_tokens}")
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            self.console.print(f"\n\n[bold red]Error:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate infinite tokens from a language model")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Model name to use from Hugging Face")
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                        help="Initial prompt to start generation")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--delay", type=float, default=0.05, 
                        help="Delay between token display in seconds")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize and run the generator
    generator = InfiniteTokenGenerator(model_name=args.model, device=args.device)
    generator.generate_infinite_tokens(
        initial_prompt=args.prompt,
        temperature=args.temperature,
        stream_delay=args.delay
    )

if __name__ == "__main__":
    main() 