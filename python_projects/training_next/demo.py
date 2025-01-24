import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from rich.console import Console
from rich.table import Table
from training_next.model_trainer import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import PreTrainedTokenizerFast
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
console = Console()

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

def calculate_loss(logits: torch.Tensor, selected_token_id: int) -> float:
    """Calculate negative log probability of the selected token."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs[selected_token_id].item()

def generate_with_probabilities(model, tokenizer, prompt: str, max_length: int = 50, top_k: int = 5):
    """Generate text and show token probabilities at each step."""
    logger.info(f"Generating from prompt: {prompt}")
    
    # Store all steps for final table
    all_steps = []
    generated_tokens = []
    token_probs = []
    entropies = []
    losses = []
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    for step in range(max_length):
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Calculate entropy
            entropy = calculate_entropy(next_token_logits)
            entropies.append(entropy)
            
            # Get top K tokens and their probabilities
            top_tokens = get_top_n_tokens(tokenizer, next_token_logits, top_k)
            all_steps.append(top_tokens)
            
            # Select the top token
            selected_token_id = torch.argmax(next_token_logits)
            selected_token = tokenizer.decode([selected_token_id.item()]).replace('Ġ', ' ').strip()
            selected_prob = F.softmax(next_token_logits, dim=-1)[selected_token_id].item()
            
            # Calculate loss (negative log probability of selected token)
            loss = calculate_loss(next_token_logits, selected_token_id.item())
            losses.append(loss)
            
            # Store token and probability
            generated_tokens.append(selected_token)
            token_probs.append(selected_prob)
            
            # Update input_ids
            input_ids = torch.cat([input_ids, selected_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
            
            # Stop if we generate an end token
            if selected_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Create and display the probability table
    table = Table(title=f"Token Probabilities by Step", show_lines=True)
    table.add_column("Step", justify="right", style="cyan", width=4)
    for i in range(top_k):
        table.add_column(f"#{i+1}", justify="left", style="magenta", width=8)
        table.add_column(f"P{i+1}", justify="right", style="green", width=6)
    table.add_column("Entropy", justify="right", style="yellow", width=6)
    table.add_column("Loss", justify="right", style="red", width=6)
    
    # Add rows for each step
    for step, (tokens, entropy, loss) in enumerate(zip(all_steps, entropies, losses), 1):
        row = [str(step)]
        for token, prob in tokens:
            row.extend([f"'{token}'", f"{prob:0.4f}"])
        row.extend([f"{entropy:0.4f}", f"{loss:0.4f}"])
        table.add_row(*row)
    
    console.print(table)
    
    # Show final generated text
    final_text = prompt + ''.join(f" {token}" for token in generated_tokens)
    console.print("\n[bold green]Final generated text:[/bold green]")
    console.print(final_text)
    
    # Show generation statistics
    avg_prob = sum(token_probs) / len(token_probs)
    min_prob = min(token_probs)
    max_prob = max(token_probs)
    avg_entropy = sum(entropies) / len(entropies)
    avg_loss = sum(losses) / len(losses)
    
    stats_table = Table(title="Generation Statistics", show_lines=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="green")
    stats_table.add_row("Average probability", f"{avg_prob:0.4f}")
    stats_table.add_row("Min probability", f"{min_prob:0.4f}")
    stats_table.add_row("Max probability", f"{max_prob:0.4f}")
    stats_table.add_row("Average entropy", f"{avg_entropy:0.4f}")
    stats_table.add_row("Average loss", f"{avg_loss:0.4f}")
    console.print("\n")
    console.print(stats_table)
    
    return final_text, token_probs, entropies, losses

def main():
    try:
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(MODEL_DIR))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
        model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
        
        # Test prompts
        prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In a world where"
        ]
        
        for prompt in prompts:
            console.print(f"\n[bold]===== Generating from prompt: {prompt} =====[/bold]")
            generate_with_probabilities(model, tokenizer, prompt, max_length=20)
            console.print("\n" + "="*50 + "\n")
            
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
    finally:
        logger.info("Demo completed")

if __name__ == "__main__":
    main()
