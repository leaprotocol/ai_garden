import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import logging
from rich.console import Console
from rich.table import Table
import os
from dotenv import load_dotenv
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass
from queue import PriorityQueue

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class BeamHypothesis:
    tokens: torch.Tensor
    score: float
    finished: bool = False

def beam_search(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    num_beams: int = 5,
    max_length: int = 50,
    temperature: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Implement beam search with detailed logging and visualization.
    """
    device = model.device
    batch_size = input_ids.shape[0]
    vocab_size = model.config.vocab_size
    
    # Initialize beams with the input sequence
    beams = [BeamHypothesis(input_ids[0], 0.0)]
    finished_beams = []
    
    logger.debug(f"Starting beam search with {num_beams} beams")
    
    for step in range(max_length):
        logger.debug(f"Step {step+1}/{max_length}")
        if len(finished_beams) >= num_beams:
            logger.debug("All beams finished, exiting early.")
            break
            
        candidates = PriorityQueue()
        
        # Expand each beam
        for beam_idx, beam in enumerate(beams):
            if beam.finished:
                logger.debug(f"  Beam {beam_idx+1} already finished, skipping.")
                continue
                
            logger.debug(f"  Expanding beam {beam_idx+1}: {tokenizer.decode(beam.tokens, skip_special_tokens=True)}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(beam.tokens.unsqueeze(0))
                logits = outputs.logits[:, -1, :] / temperature
                
            # Get top-k probabilities and tokens
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], num_beams)
            
            # Add candidates to priority queue
            for prob, token_id in zip(top_probs, top_indices):
                new_tokens = torch.cat([beam.tokens, token_id.unsqueeze(0)])
                score = beam.score - torch.log(prob).item()  # Negative log probability
                
                logger.debug(f"    Candidate: {tokenizer.decode(new_tokens, skip_special_tokens=True)}, score: {score:.4f}")
                
                # Check if sequence should end
                if token_id.item() == tokenizer.eos_token_id:
                    logger.debug(f"      EOS token found, adding to finished beams.")
                    finished_beams.append((new_tokens, score))
                else:
                    candidates.put((score, len(candidates.queue), BeamHypothesis(new_tokens, score)))
        
        # Select top-k candidates as new beams
        beams = []
        for _ in range(min(num_beams - len(finished_beams), candidates.qsize())):
            score, _, hypothesis = candidates.get()
            beams.append(hypothesis)
            logger.debug(f"  Selected beam: {tokenizer.decode(hypothesis.tokens, skip_special_tokens=True)}, score: {score:.4f}")
    
    # Add unfinished beams to finished list
    logger.debug("Adding unfinished beams to finished list.")
    finished_beams.extend([(beam.tokens, beam.score) for beam in beams])
    
    # Sort and decode results
    logger.debug("Sorting and decoding results.")
    finished_beams.sort(key=lambda x: x[1])
    results = []
    for tokens, score in finished_beams[:num_beams]:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append((text, score))
    
    return results

# Set HF token from environment
hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
if not hf_token:
    logger.warning("No Hugging Face token found in .env file!")

# Model and dataset details
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "O1-OPEN/OpenO1-SFT"

# Set token for model loading
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

logger.info(f"Attempting to load model {MODEL_NAME}")

def main():
    """Loads dataset, model, and performs beam search, displaying results."""
    try:
        # Load model and tokenizer (using your existing code)
        logger.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to("cpu")
        model.eval()
        logger.info("Model loaded successfully.")

        # Load dataset (using your existing code)
        logger.info(f"Loading dataset: {DATASET_NAME}")
        dataset = load_dataset(DATASET_NAME, split="train")
        logger.info("Dataset loaded successfully.")

        # Process a few examples with beam search
        num_examples = 3
        for i in range(num_examples):
            # Get input text
            line = dataset[i]['output']
            if not line:
                logger.warning(f"Skipping empty line {i+1}.")
                continue
            
            logger.info(f"Processing example {i+1}: {line[:100]}...")

            # Tokenize input
            input_ids = tokenizer(line, return_tensors="pt", truncation=True, max_length=20).input_ids
            input_ids = input_ids.to(model.device)

            # Perform beam search
            beam_results = beam_search(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                num_beams=5,
                max_length=20,
                temperature=0.7
            )

            # Display results
            table = Table(title=f"Beam Search Results for Example {i+1}")
            table.add_column("Beam", justify="right", style="cyan")
            table.add_column("Text", style="magenta")
            table.add_column("Score", justify="right", style="green")

            for beam_idx, (text, score) in enumerate(beam_results, 1):
                table.add_row(
                    f"#{beam_idx}",
                    text,
                    f"{score:.4f}"
                )

            console.print(table)
            console.print("\n")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 