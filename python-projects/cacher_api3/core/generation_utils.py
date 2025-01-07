import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from rich import print
from rich.table import Table
from rich.console import Console
import logging
import time

# Initialize Rich Console
console = Console()
logger = logging.getLogger(__name__)


def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top N tokens and their probabilities from logits.
    
    Args:
        tokenizer: The tokenizer to decode token IDs
        logits: The model's output logits for next token prediction
        n: Number of top tokens to return
    
    Returns:
        List of tuples containing (token_text, probability)
    """
    probabilities = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=n)
    
    return [(tokenizer.decode(idx.item()).strip(), prob.item()) 
            for idx, prob in zip(top_indices, top_probs)]


def calculate_entropy(probabilities: torch.Tensor) -> float:
    """
    Calculate the entropy of the probability distribution.
    
    Args:
        probabilities: A tensor of probabilities.
    
    Returns:
        Entropy value.
    """
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
    return entropy.item()


def visualize_next_token_sequences(
    tokenizer,
    model,
    sentence: str,
    max_tokens: int = 10,
    top_n_best: int = 10,
    top_k_beam_width: int = 5,
    temperature: float = 1.0,
    reverse: bool = False
):
    """
    Visualizes the next token sequences using beam search.
    
    Args:
        tokenizer: The tokenizer for the model.
        model: The language model.
        sentence: The input sentence.
        max_tokens: Maximum number of tokens to generate.
        top_n_best: Number of top tokens to display.
        top_k_beam_width: Beam width for beam search.
        temperature: Sampling temperature.
        reverse: Whether to reverse the input sentence.
    """
    if reverse:
        sentence = sentence[::-1]
    
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    
    table = Table(title=f"Next Token Sequence Analysis for: '{sentence}'")
    table.add_column("Step", style="cyan", justify="right")
    table.add_column("Sequence", style="magenta")
    table.add_column("Token IDs", style="blue")
    table.add_column("Completion", style="green")
    table.add_column("Probability", style="yellow")
    table.add_column("Length", style="white")
    
    
    end_token_ids = set()
    special_token_ids = set()

    # Add model's special tokens
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        end_token_ids.add(tokenizer.eos_token_id)
        special_token_ids.add(tokenizer.eos_token_id)
    
    
    def beam_search(
        input_ids: torch.Tensor,
        max_tokens: int,
        top_k_beam_width: int,
        temperature: float
    ) -> List[Dict[str, Any]]:
        """
        Performs beam search to generate sequences.
        
        Args:
            input_ids: Input token IDs.
            max_tokens: Maximum number of tokens to generate.
            top_k_beam_width: Beam width.
            temperature: Sampling temperature.
        
        Returns:
            List of beam search results.
        """
        
        sequences = [{"ids": input_ids, "score": 0.0, "text": tokenizer.decode(input_ids[0], skip_special_tokens=True), "completed": False}]
        
        for step in range(max_tokens):
            new_sequences = []
            for seq in sequences:
                if seq["completed"]:
                    new_sequences.append(seq)
                    continue
                
                current_ids = seq["ids"]
                with torch.no_grad():
                    outputs = model(current_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Get top k tokens
                    top_k_probs, top_k_indices = torch.topk(F.softmax(next_token_logits, dim=-1), k=top_k_beam_width, dim=-1)
                    
                    for i in range(top_k_beam_width):
                        next_token_id = top_k_indices[0, i].unsqueeze(0).unsqueeze(0)
                        next_token_prob = top_k_probs[0, i].item()
                        
                        new_ids = torch.cat([current_ids, next_token_id], dim=1)
                        new_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
                        new_score = seq["score"] + torch.log(top_k_probs[0, i]).item()
                        
                        completed = False
                        if next_token_id.item() in end_token_ids:
                            completed = True
                        
                        new_sequences.append({
                            "ids": new_ids,
                            "score": new_score,
                            "text": new_text,
                            "completed": completed
                        })
            
            # Sort by score and keep top k
            sequences = sorted(new_sequences, key=lambda x: x["score"], reverse=True)[:top_k_beam_width]
        
        return sequences
    
    beam_results = beam_search(input_ids, max_tokens, top_k_beam_width, temperature)
    
    for step, seq in enumerate(beam_results):
        
        ids = seq["ids"].tolist()[0]
        text = seq["text"]
        completed = seq["completed"]
        score = seq["score"]
        
        table.add_row(
            str(step + 1),
            text,
            str(ids),
            "[bold green]✓[/bold green]" if completed else "",
            f"{score:.4f}",
            str(len(ids))
        )
    
    console.print(table)


def generate_text(
    model,
    tokenizer,
    input_text=None,
    input_ids=None,
    attention_mask=None,
    seed=None,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    past_key_values=None,
    use_cache=False
):
    """
    Generates text with optional cached state.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        input_text (str, optional): The input text prompt.
        input_ids (torch.Tensor, optional): Pre-tokenized input IDs.
        attention_mask (torch.Tensor, optional): Attention mask for input IDs.
        seed (int, optional): Seed for reproducibility.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        past_key_values (tuple, optional): Cached past key values.
        use_cache (bool): Whether to use cache.

    Returns:
        Dict containing:
            - text: The generated text
            - input_ids: Updated input IDs
            - attention_mask: Updated attention mask
            - past_key_values: Updated cache (if use_cache is True)
    """
    start_time = time.time()
    
    # Set seed for reproducibility if provided
    if seed is not None:
        logger.debug(f"Setting seed to {seed}")
        from transformers import set_seed
        set_seed(seed)
        
        # Reset model's internal state
        if hasattr(model, 'reset_state'):
            model.reset_state()
        elif hasattr(model, 'init_cache'):
            model.init_cache()
    
    if input_text is not None:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
    elif input_ids is not None and attention_mask is not None:
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
    else:
        raise ValueError("Either input_text or input_ids and attention_mask must be provided.")
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # Enable sampling if temperature or top_p is set
        "do_sample": True if temperature > 0 or top_p < 1.0 else False,
        "use_cache": use_cache,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 0,  # Disable n-gram repetition blocking
        "repetition_penalty": 1.0,  # Disable repetition penalty
        "length_penalty": 1.0,  # Neutral length penalty
        "early_stopping": False  # Don't stop early
    }
    
    if past_key_values is not None:
        generation_kwargs["past_key_values"] = past_key_values
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        # outputs is now a tensor of shape [batch_size, sequence_length]
        if hasattr(outputs, 'sequences'):
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            new_attention_mask = torch.ones_like(outputs.sequences)
            new_input_ids = outputs.sequences
        else:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_attention_mask = torch.ones_like(outputs)
            new_input_ids = outputs
        
        # Extract past_key_values if available
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            past_key_values_out = outputs.past_key_values
        elif hasattr(outputs, 'cache') and outputs.cache is not None:
            past_key_values_out = outputs.cache
        else:
            past_key_values_out = None
        
        return {
            "text": generated_text,
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "past_key_values": past_key_values_out
        } 