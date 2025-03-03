import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.progress import track
import logging
from typing import Dict, Optional
from pathlib import Path
import json
from datetime import datetime

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Main log file for general info
log_file = log_dir / f"lambada_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# Detailed results log file
details_log_file = log_dir / f"lambada_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure main logger
logging.basicConfig(
    level=logging.INFO,  # Changed back to INFO
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure details logger
details_logger = logging.getLogger('details')
details_handler = logging.FileHandler(details_log_file)
details_handler.setFormatter(logging.Formatter('%(message)s'))
details_logger.addHandler(details_handler)
details_logger.propagate = False  # Prevent messages from going to root logger

def load_model(model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct") -> tuple:
    """Load model and tokenizer."""
    logger.info(f"Loading tokenizer for: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info(f"Model loaded on device: {device}")
    
    return model, tokenizer, device

def get_token_probabilities(logits: torch.Tensor, tokenizer, target_token: str, context: str, 
                          eval_model, eval_tokenizer, device) -> Dict[str, float]:
    """Get probabilities for the target token, handling tokenization correctly."""
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get the correct tokenization of the target
    full_text = context + " " + target_token
    context_ids = tokenizer(context, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    target_ids = full_ids[len(context_ids):].tolist()
    
    # Log tokenization details
    logger.debug(f"Target word: '{target_token}' -> tokens: {[tokenizer.decode([id]) for id in target_ids]}")
    
    # Get top predictions
    top_k = 10
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    top_tokens = [(tokenizer.decode([idx.item()]).strip(), prob.item()) 
                  for idx, prob in zip(top_indices, top_probs)]
    
    # Calculate metrics
    target_prob = sum(probabilities[id].item() for id in target_ids) / len(target_ids)
    
    target_rank = None
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    for id in target_ids:
        for rank, idx in enumerate(sorted_indices):
            if idx.item() == id:
                target_rank = rank + 1 if target_rank is None else min(target_rank, rank + 1)
                break
    
    target_in_top = any(id in top_indices for id in target_ids)
            
    # Evaluate top predictions with BERT
    bert_scores = {}
    for token, _ in top_tokens[:5]:  # Evaluate top 5 predictions
        bert_scores[token] = evaluate_token_validity(context, token, eval_model, eval_tokenizer, device)
    
    # Calculate relative validity score
    target_bert_score = evaluate_token_validity(context, target_token, eval_model, eval_tokenizer, device)
    relative_scores = {token: score / target_bert_score for token, score in bert_scores.items()}
    
    return {
        'target_prob': target_prob,
        'target_rank': target_rank,
        'top_predictions': top_tokens,
        'target_in_top_10': target_in_top,
        'target_ids': target_ids,
        'target_tokens': [tokenizer.decode([id]) for id in target_ids],
        'bert_scores': bert_scores,
        'target_bert_score': target_bert_score,
        'relative_bert_scores': relative_scores
    }

def predict_next_token(model, tokenizer, text: str, device: str) -> Optional[Dict]:
    """Predict the next token for a given text."""
    try:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for last position
            
        return logits
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

def load_eval_model():
    eval_model_name = "bert-base-uncased"
    logger.info(f"Loading evaluation model: {eval_model_name}")
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    eval_model = AutoModelForSequenceClassification.from_pretrained(eval_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = eval_model.to(device).eval()
    return eval_model, eval_tokenizer, device

def evaluate_token_validity(context: str, token: str, eval_model, eval_tokenizer, device) -> float:
    """
    Evaluate how well a token fits in the context using BERT.
    Returns a score between 0 and 1 where higher is better.
    """
    # Prepare input
    text = context + " " + token
    inputs = eval_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    # Get model output
    with torch.no_grad():
        outputs = eval_model(**inputs)
        logits = outputs.logits
        # Convert logits to probability using softmax
        probs = torch.softmax(logits, dim=-1)
        # Use the probability of the positive class as score
        score = probs[0][1].item()
    
    return score

def main():
    try:
        # Load main model
        model, tokenizer, device = load_model()
        
        # Load evaluation model
        eval_model, eval_tokenizer, eval_device = load_eval_model()
        
        # Load LAMBADA dataset
        SAMPLE_SIZE = 100
        logger.info(f"Loading LAMBADA dataset (first {SAMPLE_SIZE} examples)")
        dataset = load_dataset("lambada", split=f"validation[:{SAMPLE_SIZE}]")
        
        # Track metrics
        metrics = {
            'total_examples': 0,
            'correct_predictions': 0,
            'total_target_prob': 0.0,
            'ranks': [],
            'detailed_results': []  # Store detailed results for each example
        }
        
        # Process examples with progress bar
        for i in track(range(len(dataset)), description="Processing examples"):
            example = dataset[i]
            text = example['text']
            words = text.split()
            target = words[-1]
            context = ' '.join(words[:-1])
            
            # Get predictions
            logits = predict_next_token(model, tokenizer, context, device)
            if logits is None:
                continue
                
            results = get_token_probabilities(logits, tokenizer, target, context, 
                                              eval_model, eval_tokenizer, eval_device)
            
            # Store detailed results
            example_result = {
                'example_id': i + 1,
                'target': target,
                'target_tokens': results['target_tokens'],
                'target_prob': results['target_prob'],
                'target_rank': results['target_rank'],
                'top_predictions': results['top_predictions'],
                'success': results['target_in_top_10']
            }
            metrics['detailed_results'].append(example_result)
            
            # Update metrics
            metrics['total_examples'] += 1
            metrics['correct_predictions'] += int(results['target_in_top_10'])
            metrics['total_target_prob'] += results['target_prob']
            if results['target_rank']:
                metrics['ranks'].append(results['target_rank'])
            
            # Log one-line info to details file
            top_pred = results['top_predictions'][0][0] if results['top_predictions'] else 'N/A'
            
            # Format top 5 predictions with their probabilities and BERT scores
            top_5_formatted = " | ".join(
                f"'{token}'({prob*100:.1f}%,B:{results['bert_scores'].get(token, 0):.2f})"
                for token, prob in results['top_predictions'][:5]
            )
            
            # First line: metrics
            details_logger.info(
                f"Ex {i+1:4d} | "
                f"Target: '{target}' ({','.join(results['target_tokens'])}) [B:{results['target_bert_score']:.2f}] | "
                f"Top5: {top_5_formatted} | "
                f"Rank: {results['target_rank']} | "
                f"Prob: {results['target_prob']*100:.2f}% | "
                f"Success: {'✓' if results['target_in_top_10'] else '✗'}"
            )
            
            # Second line: full text
            details_logger.info(f"Full text: {text}")
            
            # Log progress every 100 examples to main log
            if (i + 1) % 100 == 0:
                success_rate = metrics['correct_predictions'] / metrics['total_examples'] * 100
                logger.info(f"Processed {i+1} examples. Current success rate: {success_rate:.1f}%")
        
        # Calculate final metrics
        final_metrics = {
            'total_examples': metrics['total_examples'],
            'success_rate': metrics['correct_predictions'] / metrics['total_examples'] * 100,
            'avg_target_prob': metrics['total_target_prob'] / metrics['total_examples'] * 100,
            'median_rank': sorted(metrics['ranks'])[len(metrics['ranks'])//2] if metrics['ranks'] else None
        }
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"lambada_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': final_metrics,
                'detailed_results': metrics['detailed_results']
            }, f, indent=2)
        
        # Display final metrics
        print("\n[bold green]Final Results[/bold green]")
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        
        metrics_table.add_row("Total Examples", str(final_metrics['total_examples']))
        metrics_table.add_row("Success Rate", f"{final_metrics['success_rate']:.1f}%")
        metrics_table.add_row("Average Target Probability", f"{final_metrics['avg_target_prob']:.2f}%")
        if final_metrics['median_rank']:
            metrics_table.add_row("Median Rank", str(final_metrics['median_rank']))
        
        print(metrics_table)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        logger.info("Analysis completed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 