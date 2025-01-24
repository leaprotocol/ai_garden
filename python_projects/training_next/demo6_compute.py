import logging
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import json
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def calculate_entropy(probs: torch.Tensor) -> float:
    """Calculate entropy of a probability distribution."""
    # Filter out zero probabilities to avoid log(0)
    non_zero_probs = probs[probs > 0]
    return -torch.sum(non_zero_probs * torch.log2(non_zero_probs)).item()

def calculate_loss(prob: float, epsilon: float = 1e-10) -> float:
    """Calculate negative log likelihood loss for a probability."""
    # Add small epsilon to avoid log(0)
    return -torch.log2(torch.tensor(prob + epsilon)).item()

def predict_masked_token(model, tokenizer, text: str) -> tuple:
    """Predict the masked token and return its probability and entropy."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_token_index, :]
        probs = torch.softmax(logits, dim=-1)
        
        return probs[0], calculate_entropy(probs[0])

def predict_next_token(model, tokenizer, current_inputs, num_tokens=5):
    """Predict next tokens using SmolLM model.
    
    Args:
        model: The SmolLM model
        tokenizer: The SmolLM tokenizer
        current_inputs: Current input tokens
        num_tokens: Number of tokens to generate ahead
        
    Returns:
        List of (token, probability) pairs for top predictions
    """
    # Ensure we have a valid input tensor
    if current_inputs.shape[1] == 0:
        # If empty, start with a space token
        input_ids = torch.tensor([[tokenizer.encode(" ", add_special_tokens=False)[0]]], device=model.device)
    else:
        input_ids = current_inputs
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    logging.debug(f"SmolLM input tokens: {[tokenizer.decode(t) for t in input_ids[0]]}")
    
    with torch.no_grad():
        # Get logits for next token prediction
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[0, -1, :]  # Get logits for last position
        
        # Get probabilities and top predictions
        probs = torch.softmax(next_token_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, k=num_tokens)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode(idx)
            predictions.append((token, prob.item()))
            logging.debug(f"SmolLM prediction: '{token}' with prob {prob.item():.4f}")
        
        return predictions

def get_token_probability(probs: torch.Tensor, tokenizer, token: str) -> float:
    """Get probability of a specific token."""
    token_id = tokenizer.encode(token, add_special_tokens=False)[0]
    return probs[token_id].item()

def get_neighbor_probability(model, tokenizer, token: str, left_neighbor: str = None, right_neighbor: str = None) -> tuple:
    """Get probability of a token with just one neighbor and the entropy."""
    if left_neighbor:
        text = f"{left_neighbor} [MASK]"
    elif right_neighbor:
        text = f"[MASK] {right_neighbor}"
    else:
        return 0.0, 0.0
        
    probs, entropy = predict_masked_token(model, tokenizer, text)
    prob = get_token_probability(probs, tokenizer, token)
    return prob, entropy

def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 5) -> list:
    """Get the top N tokens and their probabilities from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=n)
    return [(tokenizer.decode([idx.item()]).strip(), prob.item()) 
            for idx, prob in zip(top_indices, top_probs)]

def save_analysis_results(results, filename):
    """Save analysis results to a JSON file, converting tensors to native types."""
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(x) for x in obj]
        return obj

    # Convert results to JSON-serializable format
    serializable_results = convert_to_serializable(results)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    logging.info(f"Saving results to {filename}")
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_analysis_results(input_file: Path) -> dict:
    """Load analysis results from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

class LlamaTokenizerWrapper:
    def __init__(self, llama_model):
        self.model = llama_model

    def encode(self, text, add_special_tokens=True):
        # Convert text to token IDs using llama_cpp's tokenize
        tokens = self.model.tokenize(text.encode('utf-8'))
        logging.debug(f"Encoded tokens: {tokens}")
        return tokens

    def decode(self, token_ids):
        # Convert token IDs back to text
        if isinstance(token_ids, int):
            decoded = self.model.token_to_str(token_ids)
        else:
            decoded = ''.join([self.model.token_to_str(tid) for tid in token_ids])
        logging.debug(f"Decoded text: {decoded}")
        return decoded

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        encoded = self.encode(text)
        if return_tensors == "pt":
            encoded = {"input_ids": torch.tensor(encoded, dtype=torch.long)}
        return encoded

def validate_probabilities(probs: torch.Tensor, token_id: int, token_text: str) -> None:
    """Validate probability computation and log warnings for suspicious values."""
    prob = probs[token_id].item()
    
    # Check if probability is suspiciously low
    if prob < 0.01:
        logging.warning(f"Very low probability ({prob:.4f}) for token '{token_text}'")
        
        # Log top predictions for context
        top_probs, top_indices = torch.topk(probs, k=5)
        logging.warning("Top predictions were:")
        for p, idx in zip(top_probs, top_indices):
            logging.warning(f"  {idx.item()}: {p.item():.4f}")
    
    # Check if probability distribution is valid
    total_prob = torch.sum(probs).item()
    if not (0.99 < total_prob < 1.01):
        logging.error(f"Invalid probability distribution (sum = {total_prob:.4f})")

def main():
    try:
        # Initialize BERT
        logger.info("Starting analysis...")
        BERT_MODEL_NAME = "prajjwal1/bert-small"
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)
        
        # Initialize SmolLM
        SMOL_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
        logger.info(f"Loading SmolLM from {SMOL_MODEL_NAME}...")
        smol_tokenizer = AutoTokenizer.from_pretrained(SMOL_MODEL_NAME)
        smol_model = AutoModelForCausalLM.from_pretrained(
            SMOL_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create analysis directory
        script_dir = Path(__file__).parent
        analysis_dir = script_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        
        # Filter valid texts - complete sentences or paragraphs
        valid_texts = []
        for text in dataset["text"]:
            text = text.strip()
            # Skip empty lines or incomplete texts
            if not text or text.startswith('=') or text.startswith('@') or len(text.split()) < 5:
                continue
            # Skip texts that start with dots or other incomplete markers
            if text.startswith('..') or text.startswith(',,'):
                continue
            valid_texts.append(text)
        
        SAMPLE_SIZE = 2
        random.seed(42)  # For reproducibility
        examples = random.sample(valid_texts, SAMPLE_SIZE)
        
        # Log selected examples
        for i, example in enumerate(examples):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Length: {len(example.split())} words")
            logger.info(f"Text: {example}\n")
        
        # Compute analysis
        logger.info("Computing analysis...")
        results = compute_analysis(bert_model, bert_tokenizer, smol_model, smol_tokenizer, examples)
        
        # Save results
        results_file = analysis_dir / "analysis_results.json"
        logger.info(f"Saving results to {results_file}")
        save_analysis_results(results, results_file)
        
        logger.info("Analysis completed. Run visualize_analysis.py to view results.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

def compute_analysis(bert_model, bert_tokenizer, smol_model, smol_tokenizer, examples: list) -> dict:
    """Compute analysis for the given examples."""
    results = {
        "examples": [],
        "metadata": {
            "bert_model": bert_model.config.name_or_path,
            "smol_model": smol_model.config.name_or_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing sequences...", total=sum(len(bert_tokenizer.encode(ex, add_special_tokens=False)) for ex in examples))
        
        for example_idx, example in enumerate(examples):
            # Store the original text before any tokenization
            example_results = {
                "text": example,  # Store the complete original text
                "bert_forward_results": [],
                "bert_backward_results": [],
                "smol_results": [],
                "tokenization": {
                    "bert": [],
                    "smol": []
                }
            }
            
            # Tokenize the example with both tokenizers - no special tokens
            bert_tokens = bert_tokenizer.encode(example, add_special_tokens=False, return_tensors='pt')[0]
            smol_tokens = smol_tokenizer.encode(example, add_special_tokens=False)
            
            # Store tokenizations with their text representations
            for token_id in bert_tokens:
                token_text = bert_tokenizer.decode([token_id.item()])
                example_results["tokenization"]["bert"].append({
                    "id": token_id.item(),
                    "text": token_text,
                    "start_idx": example.find(token_text) if token_text in example else -1
                })
            
            for token_id in smol_tokens:
                token_text = smol_tokenizer.decode([token_id])
                example_results["tokenization"]["smol"].append({
                    "id": token_id,
                    "text": token_text,
                    "start_idx": example.find(token_text) if token_text in example else -1
                })
            
            # Get decoded tokens for debugging
            bert_decoded = [bert_tokenizer.decode(t) for t in bert_tokens]
            smol_decoded = [smol_tokenizer.decode(t) for t in smol_tokens]
            
            logging.debug(f"\nExample {example_idx+1}:")
            logging.debug(f"Original text: {example}")
            logging.debug(f"BERT decoded tokens: {bert_decoded}")
            logging.debug(f"SmolLM decoded tokens: {smol_decoded}")
            
            progress.update(task, description=f"Example {example_idx+1}/{len(examples)}: {example[:50]}...")
            
            forward_results = []
            backward_results = []
            smol_results = []
            
            # Initialize cache for SmolLM
            past_key_values = None
            
            # BERT forward analysis
            for j in range(len(bert_tokens)):
                if progress:
                    progress.update(task, advance=1)
                
                # Get the current token and its context
                current_token = bert_tokens[j].item()
                current_text = bert_tokenizer.decode([current_token])
                
                # Create masked sequence for forward prediction
                masked_tokens = bert_tokens.clone()
                masked_tokens[j] = bert_tokenizer.mask_token_id
                
                # Get full context prediction
                outputs = bert_model(**{'input_ids': masked_tokens.unsqueeze(0)})
                logits = outputs.logits[0, j]
                probs = torch.softmax(logits, dim=0)
                prob = probs[current_token].item()
                
                # Validate probabilities
                validate_probabilities(probs, current_token, current_text)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, 5)
                top_predictions = [
                    (bert_tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
                
                # Store results with top predictions
                forward_results.append({
                    "token_id": current_token,
                    "prob": prob,
                    "top_predictions": top_predictions,  # Store top N predictions
                    "entropy": calculate_entropy(probs),
                    "loss": -torch.log(probs[current_token]).item()
                })
                
                # Get neighbor-only prediction (using one token before and after if available)
                neighbor_text = ""
                if j > 0:
                    prev_token = bert_tokenizer.decode([bert_tokens[j-1]])
                    neighbor_text = prev_token + " [MASK]"
                elif j < len(bert_tokens) - 1:
                    next_token = bert_tokenizer.decode([bert_tokens[j+1]])
                    neighbor_text = "[MASK] " + next_token
                
                if neighbor_text:
                    neighbor_inputs = bert_tokenizer(neighbor_text, return_tensors='pt')
                    mask_pos = (neighbor_inputs.input_ids == bert_tokenizer.mask_token_id).nonzero()[0, 1]
                    neighbor_outputs = bert_model(**neighbor_inputs)
                    neighbor_logits = neighbor_outputs.logits[0, mask_pos]
                    neighbor_probs = torch.softmax(neighbor_logits, dim=0)
                    neighbor_prob = neighbor_probs[current_token].item()
                else:
                    neighbor_prob = 0.0
                
                # Get top backward predictions
                neighbor_top_probs, neighbor_top_indices = torch.topk(neighbor_probs, 5)
                neighbor_top_predictions = [
                    (bert_tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(neighbor_top_indices, neighbor_top_probs)
                ]
                
                backward_results.append({
                    "token_id": current_token,
                    "prob": neighbor_prob,
                    "top_predictions": neighbor_top_predictions,  # Store top N predictions
                    "entropy": calculate_entropy(neighbor_probs),
                    "loss": -torch.log(neighbor_probs[current_token]).item()
                })
                
                # Calculate entropy and loss
                entropy = calculate_entropy(probs)
                loss = calculate_loss(prob)
                
                # Log probabilities for debugging
                logging.debug(f"\nToken {j}: '{current_text}'")
                logging.debug(f"Full context prob: {prob:.4f}")
                logging.debug(f"Neighbor prob: {neighbor_prob:.4f}")
                logging.debug(f"Entropy: {entropy:.4f}")
                logging.debug(f"Loss: {loss:.4f}")
                
                # Get top predictions for display
                top_probs, top_indices = torch.topk(probs, k=5)
                predictions = []
                for p, idx in zip(top_probs, top_indices):
                    token_text = bert_tokenizer.decode([idx])
                    predictions.append((token_text, p.item()))
                    logging.debug(f"  '{token_text}': {p.item():.4f}")
                
                # SmolLM prediction at this position
                if j == 0:
                    smol_inputs = {
                        'input_ids': torch.tensor([[smol_tokens[0]]], dtype=torch.long).to(smol_model.device),
                        'attention_mask': torch.tensor([[1]], dtype=torch.long).to(smol_model.device),
                        'use_cache': True
                    }
                else:
                    current_token = smol_tokens[j-1]
                    smol_inputs = {
                        'input_ids': torch.tensor([[current_token]], dtype=torch.long).to(smol_model.device),
                        'attention_mask': torch.tensor([[1]], dtype=torch.long).to(smol_model.device),
                        'past_key_values': past_key_values,
                        'use_cache': True
                    }
                
                smol_outputs = smol_model(**smol_inputs)
                smol_logits = smol_outputs.logits[0, -1]
                smol_probs = torch.softmax(smol_logits, dim=0)
                smol_prob = smol_probs[current_token].item()
                
                # Get top SmolLM predictions
                smol_top_probs, smol_top_indices = torch.topk(smol_probs, 5)
                smol_top_predictions = [
                    (smol_tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(smol_top_indices, smol_top_probs)
                ]
                
                smol_results.append({
                    "token_id": current_token,
                    "prob": smol_prob,
                    "top_predictions": smol_top_predictions,  # Store top N predictions
                    "entropy": calculate_entropy(smol_probs),
                    "loss": -torch.log(smol_probs[current_token]).item()
                })
            
            # Store results for this example
            example_results["bert_forward_results"].extend(forward_results)
            example_results["smol_results"].extend(smol_results)
            example_results["bert_backward_results"].extend(backward_results)
            results["examples"].append(example_results)
            
            # Clear GPU memory periodically
            if torch.cuda.is_available() and example_idx % 2 == 0:
                torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    main() 