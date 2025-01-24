import torch
import logging
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path
import os

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
console = Console()

# Constants
MODEL_NAME = "prajjwal1/bert-small"  # 29M parameters
MAX_LENGTH = 128
TOP_K = 10

def get_token_probability(logits: torch.Tensor, tokenizer, token: str) -> float:
    """Get the probability of a specific token and its subwords."""
    probabilities = torch.softmax(logits, dim=-1)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    
    exact_prob = probabilities[token_ids[0]].item() if len(token_ids) == 1 else 0
    subword_probs = [probabilities[id].item() for id in token_ids]
    
    return exact_prob, max(subword_probs), sum(subword_probs) / len(subword_probs)

def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 10) -> list:
    """Get the top N tokens and their probabilities from logits."""
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=n)
    
    results = []
    for idx, prob in zip(top_indices[0], top_probs[0]):
        token = tokenizer.decode(idx.item()).strip()
        results.append((token, prob.item()))
    
    return results

def predict_masked_token(model, tokenizer, text: str, mask_position: int = -1):
    """Predict the masked token with more lenient evaluation metrics."""
    words = text.split()
    if mask_position == -1:
        mask_position = len(words) - 1
    
    original_token = words[mask_position]
    words[mask_position] = tokenizer.mask_token
    masked_text = ' '.join(words)
    
    # Group related information in log blocks
    logger.info("Input Text Analysis:")
    logger.info(f"  Original: {text}")
    logger.info(f"  Masked:   {masked_text}")
    logger.info(f"  Target:   '{original_token}' (position {mask_position})")
    
    # Tokenization debug info in a separate block
    original_token_ids = tokenizer.encode(original_token, add_special_tokens=False)
    logger.debug("Tokenization Details:")
    logger.debug(f"  Token IDs:  {original_token_ids}")
    logger.debug(f"  Decoded:    {tokenizer.decode(original_token_ids)}")
    
    inputs = tokenizer(masked_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_token_index, :]
        
        exact_prob, max_subword_prob, avg_subword_prob = get_token_probability(logits[0], tokenizer, original_token)
        
        # Log top logits in a compact format
        logger.debug("Top 10 Predictions (Raw Logits):")
        top_logits, top_indices = torch.topk(logits[0], k=10)
        for logit, idx in zip(top_logits, top_indices):
            token = tokenizer.decode(idx.item())
            logger.debug(f"  {token:15} : {logit.item():7.4f}")
    
    # Get top predictions
    top_tokens = get_top_n_tokens(tokenizer, logits, TOP_K)
    
    # Create predictions table
    table = Table(title="Masked Token Predictions", show_lines=True)
    table.add_column("Rank", justify="right", style="cyan", width=4)
    table.add_column("Predicted Token", style="magenta", width=15)
    table.add_column("Probability", justify="right", style="green", width=10)
    table.add_column("Contains Target", style="yellow", width=12)
    
    # Check for partial matches
    target_lower = original_token.lower()
    for rank, (token, prob) in enumerate(top_tokens, 1):
        contains_target = "Yes" if target_lower in token.lower() or token.lower() in target_lower else "No"
        table.add_row(
            str(rank),
            f"'{token}'",
            f"{prob:0.4f}",
            contains_target
        )
    
    console.print(table)
    
    # Show original token metrics
    metrics_table = Table(title="Target Token Metrics", show_lines=True)
    metrics_table.add_column("Metric", style="magenta", width=20)
    metrics_table.add_column("Value", justify="right", style="green", width=10)
    
    metrics_table.add_row("Exact Match Prob", f"{exact_prob:0.4f}")
    metrics_table.add_row("Best Subword Prob", f"{max_subword_prob:0.4f}")
    metrics_table.add_row("Avg Subword Prob", f"{avg_subword_prob:0.4f}")
    
    console.print(metrics_table)
    console.print("=" * 50)
    
    return {
        'top_tokens': top_tokens,
        'exact_prob': exact_prob,
        'max_subword_prob': max_subword_prob,
        'avg_subword_prob': avg_subword_prob,
        'has_partial_match': any(target_lower in t.lower() or t.lower() in target_lower for t, _ in top_tokens)
    }

def main():
    try:
        logger.info("Initializing evaluation...")
        logger.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        model.eval()
        
        logger.info("Loading LAMBADA dataset (first 20 examples)")
        from datasets import load_dataset
        dataset = load_dataset("lambada", split="validation[:20]")
        
        # Evaluation metrics
        metrics = {
            'total_examples': 0,
            'exact_matches': 0,
            'partial_matches': 0,
            'total_exact_prob': 0,
            'total_max_subword_prob': 0,
            'total_avg_subword_prob': 0
        }
        
        for i, example in enumerate(dataset, 1):
            console.print(f"\n[bold cyan]Example {i}/{len(dataset)}[/bold cyan]")
            
            text = example['text']
            words = text.split()
            context = ' '.join(words[:-1])
            target = words[-1]
            full_text = context + " " + target
            
            results = predict_masked_token(model, tokenizer, full_text)
            
            # Update metrics
            metrics['total_examples'] += 1
            metrics['exact_matches'] += any(t.lower() == target.lower() for t, _ in results['top_tokens'])
            metrics['partial_matches'] += int(results['has_partial_match'])
            metrics['total_exact_prob'] += results['exact_prob']
            metrics['total_max_subword_prob'] += results['max_subword_prob']
            metrics['total_avg_subword_prob'] += results['avg_subword_prob']
        
        # Show overall metrics
        console.print("\n[bold green]Final Results[/bold green]")
        metrics_table = Table(show_lines=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        
        metrics_table.add_row("Total Examples", str(metrics['total_examples']))
        metrics_table.add_row("Exact Matches in Top 10", str(metrics['exact_matches']))
        metrics_table.add_row("Partial Matches in Top 10", str(metrics['partial_matches']))
        metrics_table.add_row("Average Exact Prob", f"{metrics['total_exact_prob']/metrics['total_examples']:0.4f}")
        metrics_table.add_row("Average Best Subword Prob", f"{metrics['total_max_subword_prob']/metrics['total_examples']:0.4f}")
        metrics_table.add_row("Average Mean Subword Prob", f"{metrics['total_avg_subword_prob']/metrics['total_examples']:0.4f}")
        
        console.print(metrics_table)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
    finally:
        logger.info("Evaluation completed")

if __name__ == "__main__":
    main() 