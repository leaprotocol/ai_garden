import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rich.progress import track
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Main log file
log_file = log_dir / f"swag_test_{timestamp}.log"

# Separate detail files for eval and train models
eval_details_file = log_dir / f"swag_eval_{timestamp}.log"
train_details_file = log_dir / f"swag_train_{timestamp}.log"

# Configure main logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure separate loggers for eval and train details
eval_logger = logging.getLogger('eval_details')
eval_handler = logging.FileHandler(eval_details_file)
eval_handler.setFormatter(logging.Formatter('%(message)s'))
eval_logger.addHandler(eval_handler)
eval_logger.propagate = False

train_logger = logging.getLogger('train_details')
train_handler = logging.FileHandler(train_details_file)
train_handler.setFormatter(logging.Formatter('%(message)s'))
train_logger.addHandler(train_handler)
train_logger.propagate = False

def load_model(model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct") -> tuple:
    """Load model and tokenizer."""
    logger.info(f"Loading tokenizer for: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}...")
    # Load two instances - one for evaluation, one for training
    eval_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    train_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use full precision for training
        low_cpu_mem_usage=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = eval_model.to(device).eval()
    train_model = train_model.to(device).train()  # Set to training mode
    
    logger.info(f"Models loaded on device: {device}")
    
    return eval_model, train_model, tokenizer, device

def get_continuation_scores(
    model, 
    tokenizer, 
    context: str, 
    continuations: List[str],
    device: str
) -> Dict[str, float]:
    try:
        scores = []
        token_analyses = []
        position_logprobs = {}  # position -> list of logprobs
        
        # First pass: collect logprobs by position
        for continuation in continuations:
            # Tokenize and get continuation tokens
            full_text = context + " " + continuation
            inputs = tokenizer(context, return_tensors='pt').to(device)
            full_inputs = tokenizer(full_text, return_tensors='pt').to(device)
            context_len = inputs.input_ids.size(1)
            continuation_ids = full_inputs.input_ids[0, context_len:]
            
            # Process tokens
            token_probs = []
            current_context = context
            
            for i, token_id in enumerate(continuation_ids):
                # Get token probability
                current_inputs = tokenizer(current_context, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**current_inputs)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                
                token_prob = float(probs[token_id].item())
                token_logprob = float(np.log(token_prob) if token_prob > 0 else float('-inf'))
                
                # Track logprob by position
                if i not in position_logprobs:
                    position_logprobs[i] = []
                position_logprobs[i].append(token_logprob)
                
                # Store token info
                token_probs.append({
                    'token': tokenizer.decode([token_id]),
                    'probability': token_prob,
                    'logprob': token_logprob,
                    'position': i
                })
                
                current_context = tokenizer.decode(full_inputs.input_ids[0, :context_len+i+1])
            
            token_analyses.append(token_probs)
        
        # Calculate position statistics
        position_stats = {}
        for pos, logprobs in position_logprobs.items():
            valid_logprobs = [l for l in logprobs if l != float('-inf')]
            if valid_logprobs:
                position_stats[pos] = {
                    'mean': float(np.mean(valid_logprobs)),
                    'std': float(np.std(valid_logprobs)) or 1.0,  # Avoid division by zero
                    'count': len(valid_logprobs)  # Add count of valid logprobs
                }
        
        # Calculate normalized scores for each continuation
        normalized_scores = []
        for analysis in token_analyses:
            # Z-score normalize each token's logprob by position
            token_scores = []
            for token in analysis:
                pos = token['position']
                if pos in position_stats and token['logprob'] != float('-inf'):
                    stats = position_stats[pos]
                    z_score = (token['logprob'] - stats['mean']) / stats['std']
                    token_scores.append(z_score)
                else:
                    token_scores.append(float('-inf'))
            
            # Average z-scores for the sequence
            valid_scores = [s for s in token_scores if s != float('-inf')]
            score = float(np.mean(valid_scores)) if valid_scores else float('-inf')
            normalized_scores.append(score)
        
        return {
            'normalized_scores': normalized_scores,
            'token_analyses': token_analyses,
            'position_stats': position_stats
        }
        
    except Exception as e:
        logger.error(f"Score calculation failed: {e}", exc_info=True)
        return None

def get_continuation_scores_batch(
    model, 
    tokenizer, 
    contexts: List[str],
    all_continuations: List[List[str]],
    device: str
) -> List[Dict[str, float]]:
    try:
        batch_results = []
        
        # Process all continuations from all examples together
        all_position_logprobs = {}  # position -> list of logprobs
        
        for context, continuations in zip(contexts, all_continuations):
            scores = []
            token_analyses = []
            
            # Process each continuation in the batch
            for continuation in continuations:
                full_text = context + " " + continuation
                inputs = tokenizer(context, return_tensors='pt', padding=True).to(device)
                full_inputs = tokenizer(full_text, return_tensors='pt', padding=True).to(device)
                context_len = inputs['input_ids'].size(1)
                continuation_ids = full_inputs['input_ids'][0, context_len:]
                
                token_probs = []
                current_context = context
                
                # Collect all positions that need processing
                positions_to_process = []
                for i in range(len(continuation_ids)):
                    current_inputs = tokenizer(current_context, return_tensors='pt', padding=True).to(device)
                    # Store the input IDs and token ID separately
                    positions_to_process.append({
                        'input_ids': current_inputs['input_ids'],
                        'token_id': continuation_ids[i].item(),
                        'position': i
                    })
                    current_context = tokenizer.decode(full_inputs['input_ids'][0, :context_len+i+1])
                
                # Process positions in batches of 4
                for i in range(0, len(positions_to_process), 4):
                    batch_inputs = positions_to_process[i:i+4]
                    
                    # Find max sequence length in batch
                    max_len = max(inp['input_ids'].size(1) for inp in batch_inputs)
                    
                    # Pad all inputs to same length
                    padded_inputs = []
                    for inp in batch_inputs:
                        pad_size = max_len - inp['input_ids'].size(1)
                        if pad_size > 0:
                            padded = torch.cat([
                                inp['input_ids'],
                                torch.zeros((1, pad_size), dtype=torch.long, device=device)
                            ], dim=1)
                        else:
                            padded = inp['input_ids']
                        padded_inputs.append(padded)
                    
                    # Prepare batch
                    batched_inputs = torch.cat(padded_inputs, dim=0)
                    token_ids = torch.tensor([inp['token_id'] for inp in batch_inputs], device=device)
                    positions = [inp['position'] for inp in batch_inputs]
                    
                    # Get probabilities for batch
                    with torch.no_grad():
                        outputs = model(input_ids=batched_inputs)
                        logits = outputs.logits[:, -1, :]  # Last token for each sequence
                        probs = torch.softmax(logits, dim=-1)
                    
                    # Process results
                    for j, (token_id, pos) in enumerate(zip(token_ids, positions)):
                        token_prob = float(probs[j, token_id].item())
                        token_logprob = float(np.log(token_prob) if token_prob > 0 else float('-inf'))
                        
                        if pos not in all_position_logprobs:
                            all_position_logprobs[pos] = []
                        all_position_logprobs[pos].append(token_logprob)
                        
                        token_probs.append({
                            'token': tokenizer.decode([token_id]),
                            'probability': token_prob,
                            'logprob': token_logprob,
                            'position': pos
                        })
                
                token_analyses.append(token_probs)
            
            # Calculate position statistics for this example
            position_stats = {}
            for pos, logprobs in all_position_logprobs.items():
                valid_logprobs = [l for l in logprobs if l != float('-inf')]
                if valid_logprobs:
                    position_stats[pos] = {
                        'mean': float(np.mean(valid_logprobs)),
                        'std': float(np.std(valid_logprobs)) or 1.0,
                        'count': len(valid_logprobs)
                    }
            
            # Calculate normalized scores
            normalized_scores = []
            for analysis in token_analyses:
                token_scores = []
                for token in analysis:
                    pos = token['position']
                    if pos in position_stats and token['logprob'] != float('-inf'):
                        stats = position_stats[pos]
                        z_score = (token['logprob'] - stats['mean']) / stats['std']
                        token_scores.append(z_score)
                    else:
                        token_scores.append(float('-inf'))
                
                valid_scores = [s for s in token_scores if s != float('-inf')]
                score = float(np.mean(valid_scores)) if valid_scores else float('-inf')
                normalized_scores.append(score)
            
            batch_results.append({
                'normalized_scores': normalized_scores,
                'token_analyses': token_analyses,
                'position_stats': position_stats
            })
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Batch score calculation failed: {e}", exc_info=True)
        return None

def train_on_example(
    model,
    tokenizer,
    context: str,
    continuation: str,
    is_correct: bool,
    learning_rate: float,
    device: str
):
    """Train model on a single continuation."""
    # Skip training on incorrect examples
    if not is_correct:
        return
        
    # Combine context and continuation
    full_text = context + " " + continuation
    
    # Tokenize
    inputs = tokenizer(context, return_tensors='pt').to(device)
    full_inputs = tokenizer(full_text, return_tensors='pt').to(device)
    
    # Get the continuation tokens
    context_len = inputs.input_ids.size(1)
    continuation_ids = full_inputs.input_ids[0, context_len:]
    
    # First pass: find the worst token
    token_logprobs = []
    current_context = context
    
    for i, token_id in enumerate(continuation_ids):
        current_inputs = tokenizer(current_context, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**current_inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            token_prob = float(probs[token_id].item())
            token_logprob = float(np.log(token_prob) if token_prob > 0 else float('-inf'))
            token_logprobs.append((i, token_id, token_logprob))
        current_context = tokenizer.decode(full_inputs.input_ids[0, :context_len+i+1])
    
    # Find position of worst token
    worst_pos, worst_token_id, _ = min(token_logprobs, key=lambda x: x[2])
    
    # Train only on the worst token
    current_context = context
    for i in range(worst_pos + 1):
        if i < worst_pos:
            # Just update context for tokens before the worst one
            current_context = tokenizer.decode(full_inputs.input_ids[0, :context_len+i+1])
            continue
            
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Get model output for worst token position
        current_inputs = tokenizer(current_context, return_tensors='pt').to(device)
        outputs = model(**current_inputs)
        logits = outputs.logits[0, -1, :]
        
        # Create target distribution for worst token
        target_probs = torch.zeros_like(logits)
        target_probs[worst_token_id] = 1.0
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), 
            target_probs.unsqueeze(0)
        )
        
        # Backpropagate and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def calculate_sequence_score(token_analyses):
    """Calculate average logprob for a sequence."""
    logprobs = [t['logprob'] for t in token_analyses]
    valid_logprobs = [l for l in logprobs if l != float('-inf')]
    return float(np.mean(valid_logprobs)) if valid_logprobs else float('-inf')

def calculate_loss(results, correct_idx):
    """Calculate loss based on difference between correct and incorrect logprobs."""
    scores = []
    for idx, analysis in enumerate(results['token_analyses']):
        score = calculate_sequence_score(analysis)
        scores.append(score)
    
    correct_score = scores[correct_idx]
    incorrect_scores = [s for i, s in enumerate(scores) if i != correct_idx]
    avg_incorrect = float(np.mean([s for s in incorrect_scores if s != float('-inf')]))
    
    # Loss is negative if correct score is better than incorrect average
    loss = avg_incorrect - correct_score
    return loss

def main():
    try:
        # Load models
        eval_model, train_model, tokenizer, device = load_model()
        learning_rate = 3e-6  # Even smaller learning rate for fine-tuning, try 1e-7 if needed
        
        # Load SWAG dataset
        SAMPLE_SIZE = 100
        logger.info(f"Loading SWAG dataset (first {SAMPLE_SIZE} examples)")
        dataset = load_dataset("swag", "regular", split=f"validation[:{SAMPLE_SIZE}]")
        
        # Track metrics for both models
        eval_metrics = {
            'total_examples': 0,
            'correct_predictions': 0,
            'correct_by_avg_logprob': 0,
            'correct_by_median': 0,
            'correct_by_worst': 0,
            'avg_correct_prob': 0.0,
            'detailed_results': []
        }
        
        train_metrics = {
            'total_examples': 0,
            'correct_predictions': 0,
            'correct_by_avg_logprob': 0,
            'correct_by_median': 0,
            'correct_by_worst': 0,
            'avg_correct_prob': 0.0,
            'detailed_results': []
        }
        
        eval_losses = []
        train_losses = []
        
        # Process examples in batches
        BATCH_SIZE = 4
        for i in track(range(0, len(dataset), BATCH_SIZE), description="Processing examples"):
            batch_indices = range(i, min(i + BATCH_SIZE, len(dataset)))
            contexts = [dataset[j]['startphrase'] for j in batch_indices]
            all_continuations = [
                [
                    dataset[j]['ending0'],
                    dataset[j]['ending1'],
                    dataset[j]['ending2'],
                    dataset[j]['ending3']
                ] for j in batch_indices
            ]
            correct_indices = [dataset[j]['label'] for j in batch_indices]
            
            # Get scores for both models
            eval_results = get_continuation_scores_batch(eval_model, tokenizer, contexts, all_continuations, device)
            train_results = get_continuation_scores_batch(train_model, tokenizer, contexts, all_continuations, device)
            
            if eval_results is None or train_results is None:
                continue
            
            # Process each example in the batch
            for batch_idx, (eval_result, train_result, correct_idx) in enumerate(zip(eval_results, train_results, correct_indices)):
                # Update metrics for eval model
                eval_metrics['total_examples'] += 1
                eval_pred_idx = eval_result['normalized_scores'].index(max(eval_result['normalized_scores']))
                eval_avg_logprob_idx = eval_result['normalized_scores'].index(max(eval_result['normalized_scores']))
                eval_median_idx = eval_result['normalized_scores'].index(max(eval_result['normalized_scores']))
                eval_worst_idx = eval_result['normalized_scores'].index(min(eval_result['normalized_scores']))
                
                eval_metrics['correct_predictions'] += int(eval_pred_idx == correct_idx)
                eval_metrics['correct_by_avg_logprob'] += int(eval_avg_logprob_idx == correct_idx)
                eval_metrics['correct_by_median'] += int(eval_median_idx == correct_idx)
                eval_metrics['correct_by_worst'] += int(eval_worst_idx == correct_idx)
                eval_metrics['avg_correct_prob'] += eval_result['normalized_scores'][correct_idx]
                
                # Update metrics for train model
                train_metrics['total_examples'] += 1
                train_pred_idx = train_result['normalized_scores'].index(max(train_result['normalized_scores']))
                train_avg_logprob_idx = train_result['normalized_scores'].index(max(train_result['normalized_scores']))
                train_median_idx = train_result['normalized_scores'].index(max(train_result['normalized_scores']))
                train_worst_idx = train_result['normalized_scores'].index(min(train_result['normalized_scores']))
                
                train_metrics['correct_predictions'] += int(train_pred_idx == correct_idx)
                train_metrics['correct_by_avg_logprob'] += int(train_avg_logprob_idx == correct_idx)
                train_metrics['correct_by_median'] += int(train_median_idx == correct_idx)
                train_metrics['correct_by_worst'] += int(train_worst_idx == correct_idx)
                train_metrics['avg_correct_prob'] += train_result['normalized_scores'][correct_idx]
                
                # Log results
                eval_logger.info(
                    f"Ex {i*BATCH_SIZE + batch_idx+1:4d} | Correct: {correct_idx} | "
                    f"Eval(norm/avg/med/worst): {eval_pred_idx}/{eval_avg_logprob_idx}/"
                    f"{eval_median_idx}/{eval_worst_idx}"
                )
                
                eval_logger.info(f"Context: {contexts[batch_idx]}")
                
                # Log eval model predictions
                log_model_predictions(
                    eval_result, all_continuations[batch_idx], 
                    correct_idx, eval_pred_idx, eval_avg_logprob_idx, 
                    eval_median_idx, eval_worst_idx,
                    is_eval=True
                )
                
                # Log train model predictions
                train_logger.info(
                    f"Ex {i*BATCH_SIZE + batch_idx+1:4d} | Correct: {correct_idx} | "
                    f"Train(norm/avg/med/worst): {train_pred_idx}/{train_avg_logprob_idx}/"
                    f"{train_median_idx}/{train_worst_idx}"
                )
                
                train_logger.info(f"Context: {contexts[batch_idx]}")
                
                # Log train model predictions
                log_model_predictions(
                    train_result, all_continuations[batch_idx], 
                    correct_idx, train_pred_idx, train_avg_logprob_idx, 
                    train_median_idx, train_worst_idx,
                    is_eval=False
                )
                
                # Train on this example
                for idx, cont in enumerate(all_continuations[batch_idx]):
                    is_correct = (idx == correct_idx)
                    train_on_example(train_model, tokenizer, contexts[batch_idx], cont, is_correct, learning_rate, device)
                
                # Calculate losses
                eval_loss = calculate_loss(eval_result, correct_idx)
                train_loss = calculate_loss(train_result, correct_idx)
                
                eval_losses.append(eval_loss)
                train_losses.append(train_loss)
                
                # Log results with losses
                logger.info(
                    f"Ex {i*BATCH_SIZE + batch_idx+1:4d} | "
                    f"Eval loss: {eval_loss:.2f} | "
                    f"Train loss: {train_loss:.2f}"
                )
                
                # Log progress with losses
                if (i*BATCH_SIZE + batch_idx + 1) % 10 == 0:
                    avg_eval_loss = float(np.mean(eval_losses[-10:]))
                    avg_train_loss = float(np.mean(train_losses[-10:]))
                    logger.info(
                        f"Processed {i*BATCH_SIZE + batch_idx + 1} examples.\n"
                        f"  Eval model loss (last 10): {avg_eval_loss:.2f}\n"
                        f"  Train model loss (last 10): {avg_train_loss:.2f}"
                    )
        
        # Calculate and display final metrics for both models
        print("\nFinal Results:")
        print(f"Total Examples: {eval_metrics['total_examples']}")
        
        print("\nEval Model Accuracy:")
        print_model_metrics(eval_metrics)
        
        print("\nTrained Model Accuracy:")
        print_model_metrics(train_metrics)
        
        # Final loss statistics
        print("\nFinal Loss Results:")
        print(f"Eval model average loss: {float(np.mean(eval_losses)):.2f}")
        print(f"Train model average loss: {float(np.mean(train_losses)):.2f}")
        
        # Save detailed results
        save_results(eval_metrics, train_metrics)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        logger.info("Analysis completed")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def print_model_metrics(metrics):
    """Helper function to print metrics for a model."""
    total = metrics['total_examples']
    print(f"  Normalized probability: {metrics['correct_predictions'] / total * 100:.1f}%")
    print(f"  Average log probability: {metrics['correct_by_avg_logprob'] / total * 100:.1f}%")
    print(f"  Median log probability: {metrics['correct_by_median'] / total * 100:.1f}%")
    print(f"  Worst log probability: {metrics['correct_by_worst'] / total * 100:.1f}%")
    print(f"  Average probability for correct answer: {metrics['avg_correct_prob'] / total * 100:.1f}%")

def log_model_predictions(results, continuations, correct_idx, norm_idx, avg_idx, med_idx, worst_idx, is_eval=True):
    """Log predictions for either eval or train model."""
    logger_to_use = eval_logger if is_eval else train_logger
    model_name = "Eval" if is_eval else "Train"
    
    logger_to_use.info(f"{model_name} model predictions:")
    
    for idx, (cont, score, analysis) in enumerate(zip(
        continuations,
        results['normalized_scores'],
        results['token_analyses']
    )):
        marker = "→" if idx == correct_idx else " "
        pred_markers = [
            "*" if idx == norm_idx else " ",
            "+" if idx == avg_idx else " ",
            "^" if idx == med_idx else " ",
            "!" if idx == worst_idx else " "
        ]
        
        # First line: continuation and scores
        logger_to_use.info(
            f"{marker}{''.join(pred_markers)}{idx}: {cont}"
        )
        logger_to_use.info(
            f"   Token progression:"
        )
        
        # Token details
        for t in analysis:
            token = t['token'].strip()
            prob = t['probability']
            logprob = t['logprob']
            
            logger_to_use.info(
                f"     {token:15} | "
                f"p={prob:.3f} | "
                f"logp={logprob:.2f}"
            )
        
        logger_to_use.info("")  # Empty line between continuations

    # Position statistics
    logger_to_use.info(f"\n{model_name} position statistics:")
    position_stats = results['position_stats']
    for pos in sorted(position_stats.keys()):
        stats = position_stats[pos]
        logger_to_use.info(
            f"Position {pos}: mean={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}, "
            f"samples={stats['count']}"
        )
    logger_to_use.info("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()