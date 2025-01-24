import logging
from rich.console import Console
from rich.text import Text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import torch
import random
from collections import defaultdict
import statistics
import numpy as np
from pathlib import Path
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
console = Console()

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
    """Predict the masked token and return its probability, entropy, and loss."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_token_index, :]
        probs = torch.softmax(logits, dim=-1)
        
        return probs[0], calculate_entropy(probs[0])

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

def analyze_sequence(model, tokenizer, words: list) -> tuple:
    """Analyze a sequence from both directions, including neighbor-only probabilities, entropies, and losses."""
    forward_results = []
    backward_results = []
    
    # Forward analysis (left to right)
    for i in range(len(words)):
        prefix = words[:i+1]
        original_token = prefix[-1]
        
        # Full context
        masked_prefix = prefix[:-1] + ["[MASK]"]
        masked_text = " ".join(masked_prefix)
        probs, f_entropy = predict_masked_token(model, tokenizer, masked_text)
        full_prob = get_token_probability(probs, tokenizer, original_token)
        full_loss = calculate_loss(full_prob)
        
        # Single neighbor
        left_neighbor = words[i-1] if i > 0 else None
        neighbor_prob, n_entropy = get_neighbor_probability(model, tokenizer, original_token, left_neighbor=left_neighbor)
        neighbor_loss = calculate_loss(neighbor_prob)
        
        forward_results.append((original_token, full_prob, neighbor_prob, f_entropy, n_entropy, full_loss, neighbor_loss))
    
    # Backward analysis (right to left)
    for i in range(1, len(words) + 1):
        suffix = words[-i:]
        original_token = suffix[0]
        
        # Full context
        masked_suffix = ["[MASK]"] + suffix[1:]
        masked_text = " ".join(masked_suffix)
        probs, b_entropy = predict_masked_token(model, tokenizer, masked_text)
        full_prob = get_token_probability(probs, tokenizer, original_token)
        full_loss = calculate_loss(full_prob)
        
        # Single neighbor
        right_neighbor = words[-i+1] if i > 1 else None
        neighbor_prob, n_entropy = get_neighbor_probability(model, tokenizer, original_token, right_neighbor=right_neighbor)
        neighbor_loss = calculate_loss(neighbor_prob)
        
        backward_results.append((original_token, full_prob, neighbor_prob, b_entropy, n_entropy, full_loss, neighbor_loss))
    
    return forward_results, list(reversed(backward_results))

def get_color_style(forward_prob: float, backward_prob: float) -> str:
    """
    Generate color based on probabilities:
    - (0,0) -> White
    - (1,0) -> Pure Green
    - (0,1) -> Pure Red
    - (1,1) -> Yellow
    """
    # Ensure probabilities are in [0,1]
    forward_prob = max(0.0, min(1.0, forward_prob))
    backward_prob = max(0.0, min(1.0, backward_prob))
    
    # Base white
    red = 255
    green = 255
    blue = 255
    
    # Add pure colors based on probabilities
    red = int(255 - (forward_prob * 255) + (backward_prob * 255))  # Decrease for forward, increase for backward
    green = int(255 - (backward_prob * 255) + (forward_prob * 255))  # Decrease for backward, increase for forward
    blue = int(255 - (forward_prob + backward_prob) * 255)  # Decrease for both
    
    # Clamp values
    red = max(0, min(255, red))
    green = max(0, min(255, green))
    blue = max(0, min(255, blue))
    
    return f"rgb({red},{green},{blue})"

def get_clamped_ratio(full_prob: float, neighbor_prob: float, epsilon: float = 0.1) -> float:
    """Calculate clamped ratio between full and neighbor probabilities."""
    # Clamp probabilities to avoid division by zero and extreme ratios
    prob_a = full_prob + epsilon
    prob_b = max(epsilon,(neighbor_prob - epsilon))
    return max(prob_a / prob_b, prob_b / prob_a)


def get_ratio_color(ratio: float, is_backward: bool = False) -> str:
    """
    Generate color based on ratio:
    - ratio = 1 -> White
    - For forward:
        ratio > 1 -> More green (full context dominates)
        ratio < 1 -> More red (neighbor context dominates)
    - For backward:
        ratio > 1 -> More red (full context dominates)
        ratio < 1 -> More green (neighbor context dominates)
    """
    # Convert ratio to a normalized scale
    normalized = 1 - (1 / max(ratio, 1/ratio))  # This will be 0 for ratio=1, and approach 1 for extreme ratios
    
    # Base white
    red = 255
    green = 255
    blue = 255
    
    if is_backward:
        # Invert the color logic for backward ratios
        if ratio > 1:
            # More red for high ratios
            green = int(255 * (1 - normalized))
            blue = int(255 * (1 - normalized))
        else:
            # More green for low ratios
            red = int(255 * (1 - normalized))
            blue = int(255 * (1 - normalized))
    else:
        # Original forward logic
        if ratio > 1:
            # More green for high ratios
            red = int(255 * (1 - normalized))
            blue = int(255 * (1 - normalized))
        else:
            # More red for low ratios
            green = int(255 * (1 - normalized))
            blue = int(255 * (1 - normalized))
    
    return f"rgb({red},{green},{blue})"

def visualize_combined_results(forward_results: list, backward_results: list, tokenizer) -> None:
    """Visualize results with probabilities, ratios, entropies, and losses in a tabular format."""
    
    # Print header
    console.print("\n{:>8} {:>8} {:>6} {:>6} {:>6} {:<15} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:<30}".format(
        "F.Prob", "F.Neigh", "F.Rat", "F.Ent", "F.Loss",
        "Token",
        "B.Prob", "B.Neigh", "B.Rat", "B.Ent", "B.Loss",
        "ID", "Context"
    ))
    
    # Process each token
    for i, ((token, f_full, f_neigh, f_ent, f_n_ent, f_loss, f_n_loss), \
        (_, b_full, b_neigh, b_ent, b_n_ent, b_loss, b_n_loss)) in enumerate(zip(forward_results, backward_results)):
        
        # Get colors
        word_color = get_color_style(f_full, b_full)
        f_ratio = get_clamped_ratio(f_full, f_neigh)
        b_ratio = get_clamped_ratio(b_full, b_neigh)
        f_ratio_color = get_ratio_color(f_ratio, is_backward=False)
        b_ratio_color = get_ratio_color(b_ratio, is_backward=True)
        
        # Get actual token ID from tokenizer
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        
        # Get context (up to 3 tokens before and after)
        start_idx = max(0, i - 3)
        end_idx = min(len(forward_results), i + 4)
        context = ' '.join(t[0] for t in forward_results[start_idx:end_idx])
        if i - start_idx < 3:
            context = "... " + context
        if end_idx - i < 4:
            context = context + " ..."
        
        # Create colored text components
        text = Text()
        text.append(f"{f_full:8.2f} {f_neigh:8.2f} {f_ratio:6.1f} {f_ent:6.1f} {f_loss:6.1f} ", style=f_ratio_color)
        text.append(f"{token:<15}", style=word_color)
        text.append(f"{b_full:8.2f} {b_neigh:8.2f} {b_ratio:6.1f} {b_ent:6.1f} {b_loss:6.1f} ", style=b_ratio_color)
        text.append(f"{token_id:6d} ")
        text.append(f"{context:<30}")
        
        # Print row
        console.print(text)
    
    console.print()  # Add blank line at end

class ProbabilityStats:
    def __init__(self):
        self.forward_probs = defaultdict(list)
        self.backward_probs = defaultdict(list)
        self.all_forward_probs = []
        self.all_backward_probs = []
        self.forward_ratios = []  # Store ratios
        self.backward_ratios = []  # Store ratios
        self.forward_entropies = []  # Store entropies
        self.backward_entropies = []  # Store entropies
        
    def add_probabilities(self, word, forward_prob, backward_prob, forward_neighbor, backward_neighbor, forward_entropy, backward_entropy):
        self.forward_probs[word].append(forward_prob)
        self.backward_probs[word].append(backward_prob)
        self.all_forward_probs.append(forward_prob)
        self.all_backward_probs.append(backward_prob)
        
        # Calculate and store ratios
        self.forward_ratios.append(get_clamped_ratio(forward_prob, forward_neighbor))
        self.backward_ratios.append(get_clamped_ratio(backward_prob, backward_neighbor))
        
        # Store entropies
        self.forward_entropies.append(forward_entropy)
        self.backward_entropies.append(backward_entropy)
        
    def get_high_probability_words(self, threshold=0.5, min_occurrences=2):
        high_prob_words = defaultdict(lambda: {"forward": [], "backward": []})
        
        # Analyze forward probabilities
        for word, probs in self.forward_probs.items():
            if len(probs) >= min_occurrences:
                avg_prob = sum(probs) / len(probs)
                if avg_prob >= threshold:
                    high_prob_words[word]["forward"] = probs
                    
        # Analyze backward probabilities
        for word, probs in self.backward_probs.items():
            if len(probs) >= min_occurrences:
                avg_prob = sum(probs) / len(probs)
                if avg_prob >= threshold:
                    high_prob_words[word]["backward"] = probs
                    
        return high_prob_words
    
    def get_statistics(self):
        stats = {
            "forward": {
                "mean": statistics.mean(self.all_forward_probs),
                "median": statistics.median(self.all_forward_probs),
                "stdev": statistics.stdev(self.all_forward_probs) if len(self.all_forward_probs) > 1 else 0,
                "min": min(self.all_forward_probs),
                "max": max(self.all_forward_probs),
                "ratio_mean": statistics.mean(self.forward_ratios),
                "ratio_median": statistics.median(self.forward_ratios),
                "entropy_mean": statistics.mean(self.forward_entropies),
                "entropy_median": statistics.median(self.forward_entropies),
                "entropy_stdev": statistics.stdev(self.forward_entropies) if len(self.forward_entropies) > 1 else 0
            },
            "backward": {
                "mean": statistics.mean(self.all_backward_probs),
                "median": statistics.median(self.all_backward_probs),
                "stdev": statistics.stdev(self.all_backward_probs) if len(self.all_backward_probs) > 1 else 0,
                "min": min(self.all_backward_probs),
                "max": max(self.all_backward_probs),
                "ratio_mean": statistics.mean(self.backward_ratios),
                "ratio_median": statistics.median(self.backward_ratios),
                "entropy_mean": statistics.mean(self.backward_entropies),
                "entropy_median": statistics.median(self.backward_entropies),
                "entropy_stdev": statistics.stdev(self.backward_entropies) if len(self.backward_entropies) > 1 else 0
            }
        }
        return stats

    def get_surprising_words(self, min_occurrences=2, surprise_threshold=0.5, typical_threshold=0.2):
        """Find words that are usually low probability but have some high probability occurrences."""
        surprising_words = defaultdict(lambda: {"forward": [], "backward": []})
        
        # Analyze forward probabilities
        for word, probs in self.forward_probs.items():
            if len(probs) >= min_occurrences:
                max_prob = max(probs)
                other_probs = [p for p in probs if p != max_prob]
                if other_probs:  # Only if we have other occurrences to compare
                    avg_other = sum(other_probs) / len(other_probs)
                    if max_prob >= surprise_threshold and avg_other <= typical_threshold:
                        surprising_words[word]["forward"] = {
                            "max_prob": max_prob,
                            "avg_other": avg_other,
                            "occurrences": len(probs)
                        }
        
        # Analyze backward probabilities
        for word, probs in self.backward_probs.items():
            if len(probs) >= min_occurrences:
                max_prob = max(probs)
                other_probs = [p for p in probs if p != max_prob]
                if other_probs:  # Only if we have other occurrences to compare
                    avg_other = sum(other_probs) / len(other_probs)
                    if max_prob >= surprise_threshold and avg_other <= typical_threshold:
                        surprising_words[word]["backward"] = {
                            "max_prob": max_prob,
                            "avg_other": avg_other,
                            "occurrences": len(probs)
                        }
        
        return surprising_words

# Function to create ASCII histogram
def ascii_histogram(data, bins=10, width=50):
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    
    for count, edge in zip(hist, bin_edges):
        bar = '#' * int(width * (count / max_count))
        print(f'{edge:6.2f} | {bar}')

def output_html_results(forward_results: list, backward_results: list, tokenizer) -> str:
    """Generate HTML visualization of the analysis results."""
    html = """
    <style>
        body { font-family: 'Courier New', monospace; }
        table { border-collapse: collapse; width: 100%; }
        td, th { padding: 4px 8px; text-align: right; }
        th { border-bottom: 1px solid #ddd; }
        .token { font-weight: bold; text-align: left; }
        .context { color: #666; text-align: left; }
        tr:hover { background-color: #f5f5f5; }
    </style>
    <table>
        <tr>
            <th>F.Prob</th><th>F.Neigh</th><th>F.Rat</th><th>F.Ent</th><th>F.Loss</th>
            <th>Token</th>
            <th>B.Prob</th><th>B.Neigh</th><th>B.Rat</th><th>B.Ent</th><th>B.Loss</th>
            <th>ID</th><th>Context</th>
        </tr>
    """
    
    for i, ((token, f_full, f_neigh, f_ent, f_n_ent, f_loss, f_n_loss), \
        (_, b_full, b_neigh, b_ent, b_n_ent, b_loss, b_n_loss)) in enumerate(zip(forward_results, backward_results)):
        
        # Get token ID
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        
        # Get context
        start_idx = max(0, i - 3)
        end_idx = min(len(forward_results), i + 4)
        context = ' '.join(t[0] for t in forward_results[start_idx:end_idx])
        if i - start_idx < 3:
            context = "... " + context
        if end_idx - i < 4:
            context = context + " ..."
        
        # Calculate colors
        word_color = get_color_style(f_full, b_full)
        f_ratio = get_clamped_ratio(f_full, f_neigh)
        b_ratio = get_clamped_ratio(b_full, b_neigh)
        f_ratio_color = get_ratio_color(f_ratio, is_backward=False)
        b_ratio_color = get_ratio_color(b_ratio, is_backward=True)
        
        html += f"""
        <tr>
            <td style="color: {f_ratio_color}">{f_full:.2f}</td>
            <td style="color: {f_ratio_color}">{f_neigh:.2f}</td>
            <td style="color: {f_ratio_color}">{f_ratio:.1f}</td>
            <td style="color: {f_ratio_color}">{f_ent:.1f}</td>
            <td style="color: {f_ratio_color}">{f_loss:.1f}</td>
            <td class="token" style="color: {word_color}">{token}</td>
            <td style="color: {b_ratio_color}">{b_full:.2f}</td>
            <td style="color: {b_ratio_color}">{b_neigh:.2f}</td>
            <td style="color: {b_ratio_color}">{b_ratio:.1f}</td>
            <td style="color: {b_ratio_color}">{b_ent:.1f}</td>
            <td style="color: {b_ratio_color}">{b_loss:.1f}</td>
            <td>{token_id}</td>
            <td class="context">{context}</td>
        </tr>"""
    
    html += "\n    </table>"
    return html

def save_analysis_results(results: dict, output_file: Path) -> None:
    """Save analysis results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_analysis_results(input_file: Path) -> dict:
    """Load analysis results from a JSON file."""
    with open(input_file) as f:
        return json.load(f)

def compute_analysis(model, tokenizer, examples: list) -> dict:
    """Compute analysis for all examples without visualization or statistics.
    Only computes and stores raw probabilities, entropies, and losses."""
    results = {
        'examples': []
    }
    
    # Calculate total steps (each word in each example needs forward and backward analysis)
    total_words = sum(len(text.split()) for text in examples)
    total_steps = total_words * 2  # For both forward and backward analysis
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Create the main progress task
        task = progress.add_task("Analyzing sequences...", total=total_steps)
        
        for i, text in enumerate(examples, 1):
            words = text.split()
            if not words:
                continue
            
            # Update description to show current example
            progress.update(task, description=f"Example {i}/{len(examples)}: {text[:30]}...")
            
            # Forward analysis
            forward_results = []
            for i in range(len(words)):
                prefix = words[:i+1]
                original_token = prefix[-1]
                
                # Full context
                masked_prefix = prefix[:-1] + ["[MASK]"]
                masked_text = " ".join(masked_prefix)
                probs, f_entropy = predict_masked_token(model, tokenizer, masked_text)
                full_prob = get_token_probability(probs, tokenizer, original_token)
                full_loss = calculate_loss(full_prob)
                
                # Single neighbor
                left_neighbor = words[i-1] if i > 0 else None
                neighbor_prob, n_entropy = get_neighbor_probability(model, tokenizer, original_token, left_neighbor=left_neighbor)
                neighbor_loss = calculate_loss(neighbor_prob)
                
                forward_results.append({
                    'token': original_token,
                    'full_prob': full_prob,
                    'neighbor_prob': neighbor_prob,
                    'entropy': f_entropy,
                    'neighbor_entropy': n_entropy,
                    'loss': full_loss,
                    'neighbor_loss': neighbor_loss,
                    'token_id': tokenizer.encode(original_token, add_special_tokens=False)[0]
                })
                progress.advance(task)
            
            # Backward analysis
            backward_results = []
            for i in range(1, len(words) + 1):
                suffix = words[-i:]
                original_token = suffix[0]
                
                # Full context
                masked_suffix = ["[MASK]"] + suffix[1:]
                masked_text = " ".join(masked_suffix)
                probs, b_entropy = predict_masked_token(model, tokenizer, masked_text)
                full_prob = get_token_probability(probs, tokenizer, original_token)
                full_loss = calculate_loss(full_prob)
                
                # Single neighbor
                right_neighbor = words[-i+1] if i > 1 else None
                neighbor_prob, n_entropy = get_neighbor_probability(model, tokenizer, original_token, right_neighbor=right_neighbor)
                neighbor_loss = calculate_loss(neighbor_prob)
                
                backward_results.append({
                    'token': original_token,
                    'full_prob': full_prob,
                    'neighbor_prob': neighbor_prob,
                    'entropy': b_entropy,
                    'neighbor_entropy': n_entropy,
                    'loss': full_loss,
                    'neighbor_loss': neighbor_loss,
                    'token_id': tokenizer.encode(original_token, add_special_tokens=False)[0]
                })
                progress.advance(task)
            
            # Store example results
            example_data = {
                'text': text,
                'forward_results': forward_results,
                'backward_results': backward_results
            }
            results['examples'].append(example_data)
    
    return results

def main():
    try:
        # Initialize
        logger.info("Starting analysis...")
        MODEL_NAME = "prajjwal1/bert-small"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
        
        # Create analysis directory
        script_dir = Path(__file__).parent
        analysis_dir = script_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        
        # Get random examples
        valid_texts = [text.strip() for text in dataset["text"] if text.strip()]
        SAMPLE_SIZE = 10
        random.seed(42)  # For reproducibility
        examples = random.sample(valid_texts, SAMPLE_SIZE)
        
        # Compute analysis
        logger.info("Computing analysis...")
        results = compute_analysis(model, tokenizer, examples)
        
        # Save results
        results_file = analysis_dir / "analysis_results.json"
        logger.info(f"Saving results to {results_file}")
        save_analysis_results(results, results_file)
        
        logger.info("Analysis completed. Run visualize_analysis.py to view results.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 