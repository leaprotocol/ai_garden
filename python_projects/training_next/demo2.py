import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from rich.console import Console
from rich.table import Table
from training_next.model_trainer import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
console = Console()

# Constants
MAX_LENGTH = 128  # Maximum sequence length for the model
SMOL_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # Smol LM model name

# Get the directory where the script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = SCRIPT_DIR / "tiny_gpt"

def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 5) -> list:
    """Get the top N tokens and their probabilities from logits."""
    probabilities = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=n)
    
    return [(tokenizer.decode(idx.item()).strip().replace('Ġ', ' ').strip(), prob.item()) 
            for idx, prob in zip(top_indices, top_probs)]

def calculate_entropy(logits: torch.Tensor) -> float:
    """Calculate entropy of the probability distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs).item()

def calculate_loss(logits: torch.Tensor, target_token_id: int) -> float:
    """Calculate negative log probability of the target token."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs[target_token_id].item()

def evaluate_example(our_model, our_tokenizer, smol_model, smol_tokenizer, context: str, top_k: int = 5):
    """Evaluate and compare models on a single example."""
    logger.info(f"Context: {context}")
    
    # Get our model's prediction
    our_input_ids = our_tokenizer.encode(context, return_tensors='pt')
    if our_input_ids.shape[1] > MAX_LENGTH:
        logger.warning(f"Input sequence length ({our_input_ids.shape[1]}) exceeds maximum length ({MAX_LENGTH}). Truncating...")
        our_input_ids = our_input_ids[:, -MAX_LENGTH:]
    
    # Get Smol LM's prediction
    smol_input_ids = smol_tokenizer.encode(context, return_tensors='pt')
    
    with torch.no_grad():
        # Our model predictions
        our_outputs = our_model(our_input_ids)
        our_logits = our_outputs.logits[0, -1, :]
        our_entropy = calculate_entropy(our_logits)
        our_top_tokens = get_top_n_tokens(our_tokenizer, our_logits, top_k)
        
        # Smol LM predictions
        smol_outputs = smol_model(smol_input_ids)
        smol_logits = smol_outputs.logits[0, -1, :]
        smol_entropy = calculate_entropy(smol_logits)
        smol_top_tokens = get_top_n_tokens(smol_tokenizer, smol_logits, top_k)
        
        # Create comparison table
        table = Table(title=f"Model Predictions Comparison", show_lines=True)
        table.add_column("Rank", justify="right", style="cyan", width=4)
        table.add_column("Our Model Token", justify="left", style="magenta", width=15)
        table.add_column("Our Prob", justify="right", style="green", width=10)
        table.add_column("Smol Token", justify="left", style="magenta", width=15)
        table.add_column("Smol Prob", justify="right", style="green", width=10)
        
        # Add rows for top K predictions
        for rank, ((our_token, our_prob), (smol_token, smol_prob)) in enumerate(zip(our_top_tokens, smol_top_tokens), 1):
            table.add_row(
                str(rank),
                f"'{our_token}'",
                f"{our_prob:0.4f}",
                f"'{smol_token}'",
                f"{smol_prob:0.4f}"
            )
        
        console.print(table)
        
        # Show metrics comparison
        metrics_table = Table(title="Model Metrics Comparison", show_lines=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Our Model", justify="right", style="green")
        metrics_table.add_column("Smol LM", justify="right", style="yellow")
        metrics_table.add_row("Entropy", f"{our_entropy:0.4f}", f"{smol_entropy:0.4f}")
        
        # Calculate cross-model agreement by finding our top token in smol's predictions
        our_top_token = our_top_tokens[0][0].lower()
        smol_prob_for_our_top = 0.0
        for smol_token, smol_prob in smol_top_tokens:
            if smol_token.lower() == our_top_token:
                smol_prob_for_our_top = smol_prob
                break
        
        metrics_table.add_row(
            "Prob of our top token in Smol",
            f"{our_top_tokens[0][1]:0.4f}",
            f"{smol_prob_for_our_top:0.4f}"
        )
        
        console.print("\n")
        console.print(metrics_table)
        console.print("=" * 50)
        
        return {
            'our_entropy': our_entropy,
            'smol_entropy': smol_entropy,
            'our_top_prob': our_top_tokens[0][1],
            'smol_prob_for_our_top': smol_prob_for_our_top,
            'our_top_tokens': our_top_tokens,
            'smol_top_tokens': smol_top_tokens,
            'our_outputs': our_outputs,
            'smol_outputs': smol_outputs
        }

def main():
    try:
        logger.info("Loading models and tokenizers...")
        
        # Load our model and tokenizer
        our_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(MODEL_DIR))
        if our_tokenizer.pad_token is None:
            our_tokenizer.pad_token = our_tokenizer.eos_token
        our_model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
        our_model.eval()
        
        # Load Smol LM model and tokenizer with bfloat16
        logger.info("Loading SmolLM-135M in bfloat16 precision...")
        smol_tokenizer = AutoTokenizer.from_pretrained(SMOL_MODEL_NAME)
        smol_model = AutoModelForCausalLM.from_pretrained(
            SMOL_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        smol_model.eval()
        
        logger.info("Models and tokenizers loaded successfully")
        
        # Load LAMBADA dataset
        logger.info("Loading LAMBADA dataset...")
        dataset = load_dataset("lambada", split="validation[:100]")  # Load first 100 examples
        
        # Track overall metrics
        total_examples = 0
        total_our_entropy = 0
        total_smol_entropy = 0
        total_agreement_score = 0
        total_our_target_prob = 0
        total_smol_target_prob = 0
        
        # Evaluate each example
        for example in dataset:
            console.print(f"\n[bold]===== Example {total_examples + 1} =====[/bold]")
            
            # Split into context and target
            text = example['text']
            words = text.split()
            context = ' '.join(words[:-1])
            target = words[-1]
            
            logger.info(f"Target word: {target}")
            
            # Evaluate
            results = evaluate_example(our_model, our_tokenizer, smol_model, smol_tokenizer, context)
            
            # Update metrics
            total_examples += 1
            total_our_entropy += results['our_entropy']
            total_smol_entropy += results['smol_entropy']
            total_agreement_score += results['smol_prob_for_our_top']
            
            # Get target word probabilities by searching through all tokens
            target = target.lower()
            
            # For our model
            our_target_prob = 0.0
            our_probs = F.softmax(results['our_outputs'].logits[0, -1, :], dim=-1)
            for i in range(len(our_probs)):
                token = our_tokenizer.decode(i).strip().replace('Ġ', ' ').strip().lower()
                if token == target:
                    our_target_prob = our_probs[i].item()
                    break
            
            # For SmolLM
            smol_target_prob = 0.0
            smol_probs = F.softmax(results['smol_outputs'].logits[0, -1, :], dim=-1)
            for i in range(len(smol_probs)):
                token = smol_tokenizer.decode(i).strip().replace('Ġ', ' ').strip().lower()
                if token == target:
                    smol_target_prob = smol_probs[i].item()
                    break
            
            total_our_target_prob += our_target_prob
            total_smol_target_prob += smol_target_prob
            
            # Show target word probabilities
            target_table = Table(title="Target Word Probabilities", show_lines=True)
            target_table.add_column("Model", style="cyan")
            target_table.add_column("Probability", justify="right", style="green")
            target_table.add_row("Our Model", f"{our_target_prob:0.4f}")
            target_table.add_row("SmolLM", f"{smol_target_prob:0.4f}")
            console.print("\n")
            console.print(target_table)
            console.print("=" * 50)
        
        # Show overall results
        console.print("\n[bold]===== Overall Results =====[/bold]")
        overall_table = Table(title="Model Comparison Results", show_lines=True)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", justify="right", style="green")
        overall_table.add_row("Total examples", str(total_examples))
        overall_table.add_row("Avg Our Entropy", f"{total_our_entropy/total_examples:0.4f}")
        overall_table.add_row("Avg Smol Entropy", f"{total_smol_entropy/total_examples:0.4f}")
        overall_table.add_row("Avg Agreement Score", f"{total_agreement_score/total_examples:0.4f}")
        overall_table.add_row("Avg Our Target Prob", f"{total_our_target_prob/total_examples:0.4f}")
        overall_table.add_row("Avg Smol Target Prob", f"{total_smol_target_prob/total_examples:0.4f}")
        console.print(overall_table)
            
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
    finally:
        logger.info("Evaluation completed")

if __name__ == "__main__":
    main() 