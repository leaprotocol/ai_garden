"""
Core token generation functionality with support for both console and WebSocket output.
"""

import torch
import logging
import threading
from pathlib import Path
from rich.text import Text
from rich.console import Console
from typing import Optional, Callable, Any, List

logger = logging.getLogger(__name__)

class TokenGenerator:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        from main import InfiniteTokenGenerator
        self.generator = InfiniteTokenGenerator(model_name=model_name)
        self.console = Console()
        self.callbacks = []
        self._stop_event = threading.Event()
        
    def add_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback to receive token generation events"""
        self.callbacks.append(callback)
        
    def notify_callbacks(self, event: dict[str, Any]) -> None:
        """Notify all callbacks of an event"""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")
    
    def stop(self) -> None:
        """Stop token generation"""
        self._stop_event.set()
                
    def generate_tokens(self, prompt: str, temperature: float = 0.7):
        """Generate tokens from a prompt with both console and WebSocket output"""
        try:
            # Reset stop event
            self._stop_event.clear()
            
            # Prepare model for efficient token-by-token generation
            tokenizer = self.generator.tokenizer
            model = self.generator.model
            
            # Tokenize the initial prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.generator.device)
            attention_mask = torch.ones_like(input_ids).to(self.generator.device)
            
            # Set up for token-by-token generation
            past_key_values = None
            token_count = 0
            
            while not self._stop_event.is_set():
                # Generate a single token efficiently
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                
                # Get logits and past key values for next iteration
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature sampling
                next_token_logits = next_token_logits / temperature
                
                # Get probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Get top 3 tokens and their probabilities for display
                topk_probs, topk_indices = torch.topk(probs, k=3, dim=-1)
                
                # Decode the generated token
                new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Get token probability and color
                token_prob = probs[0, next_token.item()].item()
                prob_color = "green" if token_prob > 0.7 else "yellow" if token_prob > 0.4 else "red"
                
                # Get top tokens info
                top_tokens_info = [
                    {
                        "token": tokenizer.decode([idx.item()]),
                        "probability": prob.item()
                    }
                    for idx, prob in zip(topk_indices[0], topk_probs[0])
                ]
                
                # Create event data
                event_data = {
                    "token": new_token_text,
                    "token_probability": token_prob,
                    "probability_color": prob_color,
                    "top_tokens": top_tokens_info,
                    "token_count": token_count
                }
                
                # Notify callbacks
                self.notify_callbacks(event_data)
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.generator.device)], dim=-1)
                
                # Update token count
                token_count += 1
                
                # Manage context length to prevent OOM errors
                if input_ids.shape[1] > 2000:
                    # Keep the last 1000 tokens
                    input_ids = input_ids[:, -1000:]
                    attention_mask = attention_mask[:, -1000:]
                    # Reset past_key_values to force recomputation with the trimmed context
                    past_key_values = None
                    logger.info(f"Trimmed context to last 1000 tokens to prevent OOM")
                
                # Check for stop event more frequently
                if token_count % 5 == 0 and self._stop_event.is_set():
                    break
                
            # Notify about stopping
            logger.debug(f"Token generation stopped after {token_count} tokens")
            
            # Send stop event to callbacks
            self.notify_callbacks({
                "event": "stopped",
                "token_count": token_count
            })
                
        except Exception as e:
            logger.error(f"Error in token generation: {str(e)}")
            # Notify about error
            self.notify_callbacks({
                "event": "error",
                "error": str(e)
            })
            raise 