import logging
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
console = Console()

def find_token_span(text: str, target: str) -> tuple:
    """Find the first and last occurrence of target in text."""
    words = text.split()
    first_idx = -1
    last_idx = -1
    
    for i, word in enumerate(words):
        if word == target:
            if first_idx == -1:
                first_idx = i
            last_idx = i
    
    return first_idx, last_idx

def prepare_masked_example(text: str, target: str) -> dict:
    """Prepare text by trimming to target tokens and masking first occurrence."""
    first_idx, last_idx = find_token_span(text, target)
    if first_idx == -1 or first_idx == last_idx:
        return None
    
    words = text.split()
    # Trim to span between first and last occurrence
    trimmed_words = words[first_idx:last_idx + 1]
    # Replace first occurrence with mask
    trimmed_words[0] = "[MASK]"
    
    return {
        'original': ' '.join(words[first_idx:last_idx + 1]),
        'masked': ' '.join(trimmed_words),
        'target': target,
        'context_length': len(trimmed_words)
    }

def get_token_probability(logits: torch.Tensor, tokenizer, token: str) -> tuple:
    """Get probability of a specific token and its subwords."""
    probabilities = torch.softmax(logits, dim=-1)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    
    exact_prob = probabilities[token_ids[0]].item() if len(token_ids) == 1 else 0
    subword_probs = [probabilities[id].item() for id in token_ids]
    
    return exact_prob, max(subword_probs), sum(subword_probs) / len(subword_probs)

def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 10) -> list:
    """Get top N tokens and their probabilities."""
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=n)
    
    results = []
    for idx, prob in zip(top_indices[0], top_probs[0]):
        token = tokenizer.decode(idx.item()).strip()
        results.append((token, prob.item()))
    
    return results

def predict_masked_token(model, tokenizer, example: dict) -> dict:
    """Predict the masked token using BERT."""
    if not example:
        return None
        
    inputs = tokenizer(example['masked'], return_tensors='pt', truncation=True)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_token_index, :]
        
        # Get probabilities for target token
        exact_prob, max_subword_prob, avg_subword_prob = get_token_probability(
            logits[0], tokenizer, example['target']
        )
        
        # Get top predictions
        top_tokens = get_top_n_tokens(tokenizer, logits, 10)
        
        return {
            'top_tokens': top_tokens,
            'exact_prob': exact_prob,
            'max_subword_prob': max_subword_prob,
            'avg_subword_prob': avg_subword_prob,
            'target_in_top': any(t.lower() == example['target'].lower() for t, _ in top_tokens)
        }

def main():
    try:
        # Initialize
        logger.info("Starting reversed LAMBADA analysis...")
        MODEL_NAME = "prajjwal1/bert-small"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        model.eval()
        
        # Load dataset
        SAMPLE_SIZE = 100
        logger.info(f"Loading LAMBADA dataset (first {SAMPLE_SIZE} examples)")
        dataset = load_dataset("lambada", split=f"validation[:{SAMPLE_SIZE}]")
        
        # Analysis metrics
        metrics = {
            'total_valid': 0,
            'total_examples': 0,
            'correct_predictions': 0,
            'total_exact_prob': 0,
            'total_max_prob': 0,
            'total_avg_prob': 0
        }
        
        # Process examples
        for i, example in enumerate(dataset, 1):
            text = example['text']
            words = text.split()
            target = words[-1]  # Original LAMBADA target
            
            # Prepare masked example
            masked_example = prepare_masked_example(text, target)
            if not masked_example:
                continue
            
            metrics['total_examples'] += 1
            
            # Log example details
            if i <= 5:  # Show first 5 examples in detail
                console.print(f"\n[bold cyan]Example {i}[/bold cyan]")
                console.print(f"Original: {masked_example['original']}")
                console.print(f"Masked:   {masked_example['masked']}")
                console.print(f"Target:   {masked_example['target']}")
            
            # Get predictions
            results = predict_masked_token(model, tokenizer, masked_example)
            if not results:
                continue
                
            metrics['total_valid'] += 1
            metrics['correct_predictions'] += int(results['target_in_top'])
            metrics['total_exact_prob'] += results['exact_prob']
            metrics['total_max_prob'] += results['max_subword_prob']
            metrics['total_avg_prob'] += results['avg_subword_prob']
            
            # Show detailed results for first 5 examples
            if i <= 5:
                pred_table = Table(show_lines=True)
                pred_table.add_column("Rank", justify="right", style="cyan", width=4)
                pred_table.add_column("Token", style="magenta", width=15)
                pred_table.add_column("Probability", justify="right", style="green", width=10)
                
                for rank, (token, prob) in enumerate(results['top_tokens'], 1):
                    pred_table.add_row(str(rank), token, f"{prob:.4f}")
                
                console.print(pred_table)
        
        # Display overall results
        console.print("\n[bold green]Analysis Results[/bold green]")
        stats_table = Table(show_lines=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        if metrics['total_valid'] > 0:
            stats_table.add_row("Total Examples", str(metrics['total_examples']))
            stats_table.add_row("Valid Examples", str(metrics['total_valid']))
            stats_table.add_row("Correct in Top 10", str(metrics['correct_predictions']))
            stats_table.add_row("Success Rate", 
                              f"{100*metrics['correct_predictions']/metrics['total_valid']:.1f}%")
            stats_table.add_row("Avg Exact Prob", 
                              f"{metrics['total_exact_prob']/metrics['total_valid']:.4f}")
            stats_table.add_row("Avg Max Prob", 
                              f"{metrics['total_max_prob']/metrics['total_valid']:.4f}")
            stats_table.add_row("Avg Mean Prob", 
                              f"{metrics['total_avg_prob']/metrics['total_valid']:.4f}")
        
        console.print(stats_table)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        logger.info("Analysis completed")

if __name__ == "__main__":
    main() 