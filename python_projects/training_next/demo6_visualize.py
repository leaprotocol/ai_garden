import logging
from rich.console import Console
from pathlib import Path
import statistics
import numpy as np
from collections import defaultdict
from demo6_compute import load_analysis_results
from transformers import AutoTokenizer
from visualization_utils import (
    create_html_table, create_color_formatter, create_token_formatter,
    create_context_formatter, get_ratio_color, get_color_style,
    create_console_formatter, create_console_token_formatter, create_console_context_formatter,
    create_console_table, get_prob_color, get_entropy_color, get_loss_color,
    HTML_TEMPLATE
)
import json
from rich.table import Table
from rich.text import Text
import torch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
console = Console()

def get_clamped_ratio(full_prob: float, neighbor_prob: float, min_ratio: float = 0.1, max_ratio: float = 10.0) -> float:
    """Calculate and clamp the ratio between full and neighbor probabilities."""
    if neighbor_prob < 1e-10:  # Avoid division by zero
        return max_ratio
    ratio = full_prob / neighbor_prob
    return max(min_ratio, min(ratio, max_ratio))

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

def create_smol_predictions_column(predictions):
    """Format SmolLM predictions into a readable string."""
    return " | ".join(f"{token}({prob:.3f})" for token, prob in predictions[:3])

def get_context(forward_results, i, tokenizer):
    """Get context for a token with proper text reconstruction."""
    # Show more context - 8 tokens before and 8 after
    start_idx = max(0, i - 8)
    end_idx = min(len(forward_results), i + 9)
    
    # Get token IDs for the context window
    context_tokens = [t['token_id'] for t in forward_results[start_idx:end_idx]]
    
    # Decode the entire context sequence at once
    context_text = tokenizer.decode(context_tokens)
    
    # Add ellipsis if needed
    if start_idx > 0:
        context_text = "... " + context_text
    if end_idx < len(forward_results):
        context_text = context_text + " ..."
        
    return context_text

def align_tokens_with_positions(forward_results, backward_results, smol_results, original_text, tokenizer):
    """Align tokens with their original character positions."""
    aligned_results = []
    token_boundaries = []
    current_pos = 0
    
    # First, create token boundaries
    for i, result in enumerate(forward_results):
        token_text = tokenizer.decode([result["token_id"]]).strip()
        logging.debug(f"Processing token {i}: '{token_text}' (length: {len(token_text)})")
        logging.debug(f"  Searching for token: '{token_text}' from position: {current_pos}")
        
        # Handle special tokens and whitespace
        if token_text in tokenizer.all_special_tokens:
            start = current_pos
            end = current_pos + 1
        else:
            # Find the token in the original text
            start = original_text.find(token_text, current_pos)
            end = start + len(token_text) if start != -1 else current_pos + len(token_text)
        
        # Update current position
        current_pos = end
        
        # Store token boundaries
        token_boundaries.append({
            "token_id": result["token_id"],
            "token": token_text,
            "original_start": start,
            "original_end": end,
            "suggested_start": i,
            "suggested_end": i + 1
        })
        logging.debug(f"  Token: '{token_text}', start: {start}, end: {end}, current_pos: {current_pos}")
    
    # After creating token boundaries
    logging.debug("Token boundaries:")
    for boundary in token_boundaries:
        logging.debug(f"  Token: {boundary['token']} ({boundary['token_id']})")
        logging.debug(f"    Original: {boundary['original_start']}-{boundary['original_end']}")
        logging.debug(f"    Suggested: {boundary['suggested_start']}-{boundary['suggested_end']}")
    
    # Then create aligned results
    for char_pos in range(len(original_text)):
        relevant_tokens = [
            t for t in token_boundaries
            if t['original_start'] == char_pos or t['original_end'] == char_pos
        ]

        if relevant_tokens:
            # Create a combined result for this position
            combined = {
                'char_pos': char_pos,
                'char': original_text[char_pos],
                'tokens': relevant_tokens
            }
            aligned_results.append(combined)
            logging.debug(f"Aligned result for char_pos: {char_pos}, tokens: {[t['token'] for t in relevant_tokens]}")
    
    return aligned_results, token_boundaries

def visualize_combined_results(forward_results, backward_results, smol_results, tokenizer, original_text=None):
    """Create a table showing forward and backward analysis results."""
    logging.info("Starting HTML generation...")
    
    # Align tokens from different tokenizers
    aligned_results, token_boundaries = align_tokens_with_positions(forward_results, backward_results, smol_results, original_text, tokenizer)
    
    # Use the original text if provided, otherwise reconstruct from tokens
    if original_text:
        logging.info("Using original text from JSON")
        cleaned_text = original_text
    else:
        # Start with the full text from the token IDs
        logging.info("Decoding tokens for full text...")
        token_ids = [r['token_id'] for r in aligned_results]
        cleaned_text = tokenizer.decode(token_ids)
    
    # Create the sentence section
    logging.info(f"Creating sentence section with text length: {len(cleaned_text)}")
    sentence_section = f'<div class="sentence"><h3>Full Text:</h3><p>{cleaned_text}</p></div>\n'
    
    # Create the table
    logging.info("Creating table header...")
    table = '<table>\n<thead>\n<tr>'
    
    # Add column headers
    columns = [
        'F.Prob', 'F.Neigh', 'F.Ent', 'F.Loss', 'Token',
        'B.Prob', 'B.Ent', 'B.Loss',
        'S.Prob', 'S.Ent', 'S.Loss',
        'ID', 'Context'
    ]
    
    for col in columns:
        table += f'<th>{col}</th>'
    table += '</tr>\n</thead>\n<tbody>'
    
    # Process all tokens
    logging.info(f"Processing {len(aligned_results)} tokens for table rows...")
    for i, result in enumerate(aligned_results):
        # Get colors for each metric
        f_prob_color = get_prob_color(result['forward']['prob'], False)
        b_prob_color = get_prob_color(result['backward']['prob'] if result['backward'] else 0.0, True)
        s_prob_color = get_prob_color(result['smol']['prob'] if result['smol'] else 0.0, False)
        
        f_ent_color = get_entropy_color(result['forward']['entropy'])
        b_ent_color = get_entropy_color(result['backward']['entropy'] if result['backward'] else 0.0)
        s_ent_color = get_entropy_color(result['smol']['entropy'] if result['smol'] else 0.0)
        
        f_loss_color = get_loss_color(result['forward']['loss'])
        b_loss_color = get_loss_color(result['backward']['loss'] if result['backward'] else 0.0)
        s_loss_color = get_loss_color(result['smol']['loss'] if result['smol'] else 0.0)
        
        # Format values with proper handling of None
        b_data = result['backward'] or {'prob': 0.0, 'entropy': 0.0, 'loss': 0.0}
        s_data = result['smol'] or {'prob': 0.0, 'entropy': 0.0, 'loss': 0.0}
        
        # Get context for this token
        context = get_context(aligned_results, i, tokenizer)
        
        # Add row to table
        row = f"""
        <tr>
            <td style="color: {f_prob_color}">{result['forward']['prob']:.2f}</td>
            <td style="color: {f_prob_color}">{result['forward']['neighbor_prob']:.2f}</td>
            <td style="color: {f_ent_color}">{result['forward']['entropy']:.1f}</td>
            <td style="color: {f_loss_color}">{result['forward']['loss']:.1f}</td>
            <td class="token">{result['token']}</td>
            <td style="color: {b_prob_color}">{b_data['prob']:.2f}</td>
            <td style="color: {b_ent_color}">{b_data['entropy']:.1f}</td>
            <td style="color: {b_loss_color}">{b_data['loss']:.1f}</td>
            <td style="color: {s_prob_color}">{s_data['prob']:.6f}</td>
            <td style="color: {s_ent_color}">{s_data['entropy']:.1f}</td>
            <td style="color: {s_loss_color}">{s_data['loss']:.1f}</td>
            <td>{result['token_id']}</td>
            <td class="context">{context}</td>
        </tr>"""
        table += row
    
    table += '\n</tbody>\n</table>'
    logging.info("Table generation complete.")
    
    # Return the complete HTML using the template
    logging.info("Applying HTML template...")
    return HTML_TEMPLATE.format(content=sentence_section + table)

def create_smol_predictions_formatter():
    """Create formatter for SmolLM predictions column."""
    def formatter(value):
        return value, ""  # No special styling for now
    return formatter

def output_html_results(forward_results: list, backward_results: list, smol_results: list, tokenizer) -> str:
    """Generate HTML visualization of the analysis results."""
    
    # Prepare data in row format
    rows_data = []
    for i, ((token, f_full, f_neigh, f_ent, f_n_ent, f_loss, f_n_loss), \
        (_, b_full, b_neigh, b_ent, b_n_ent, b_loss, b_n_loss), \
        (_, s_full, s_neigh, s_ent, s_n_ent, s_loss, s_n_loss)) in enumerate(zip(forward_results, backward_results, smol_results)):
        
        # Get token ID and context
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        
        # Get context
        start_idx = max(0, i - 3)
        end_idx = min(len(forward_results), i + 4)
        context = ' '.join(t['token'] for t in forward_results[start_idx:end_idx])
        if i - start_idx < 3:
            context = "... " + context
        if end_idx - i < 4:
            context = context + " ..."
        
        # Calculate ratios
        f_ratio = get_clamped_ratio(f_full, f_neigh)
        b_ratio = get_clamped_ratio(b_full, b_neigh)
        s_ratio = get_clamped_ratio(s_full, s_neigh)
        
        rows_data.append({
            'f_prob': f_full,
            'f_neigh': f_neigh,
            'f_ratio': f_ratio,
            'f_ent': f_ent,
            'f_loss': f_loss,
            'token': (token, f_full, b_full),  # Pass both probs for coloring
            'b_prob': b_full,
            'b_neigh': b_neigh,
            'b_ratio': b_ratio,
            'b_ent': b_ent,
            'b_loss': b_loss,
            's_prob': s_full,
            's_neigh': s_neigh,
            's_ratio': s_ratio,
            's_ent': s_ent,
            's_loss': s_loss,
            'token_id': token_id,
            'context': context
        })
    
    # Define columns and their formatters
    columns = [
        'f_prob', 'f_neigh', 'f_ratio', 'f_ent', 'f_loss',
        'token',
        'b_prob', 'b_neigh', 'b_ratio', 'b_ent', 'b_loss',
        's_prob', 's_neigh', 's_ratio', 's_ent', 's_loss',
        'token_id', 'context'
    ]
    
    formatters = {
        'f_prob': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'f_neigh': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'f_ratio': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'f_ent': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'f_loss': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'token': create_token_formatter(get_color_style),
        'b_prob': create_color_formatter(lambda x: get_ratio_color(x, True)),
        'b_neigh': create_color_formatter(lambda x: get_ratio_color(x, True)),
        'b_ratio': create_color_formatter(lambda x: get_ratio_color(x, True)),
        'b_ent': create_color_formatter(lambda x: get_ratio_color(x, True)),
        'b_loss': create_color_formatter(lambda x: get_ratio_color(x, True)),
        's_prob': create_color_formatter(lambda x: get_ratio_color(x, False)),
        's_neigh': create_color_formatter(lambda x: get_ratio_color(x, False)),
        's_ratio': create_color_formatter(lambda x: get_ratio_color(x, False)),
        's_ent': create_color_formatter(lambda x: get_ratio_color(x, False)),
        's_loss': create_color_formatter(lambda x: get_ratio_color(x, False)),
        'context': create_context_formatter()
    }
    
    return create_html_table(rows_data, columns, formatters)

def get_token_style(f_prob, b_prob, s_prob):
    """Get style for token based on probabilities."""
    if f_prob < 0.01 or b_prob < 0.01:
        return "red"
    if f_prob > 0.1 or b_prob > 0.1:
        return "green"
    return "white"

def create_console_table(forward_results, backward_results, smol_results, bert_tokenizer, smol_tokenizer, console, top_n=3):
    """Create a rich console table showing tokenization differences side by side."""
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Char Pos", style="cyan")
    table.add_column("Char", style="green")
    table.add_column("BERT Token", style="yellow")
    table.add_column("BERT ID", style="blue")
    table.add_column("SmolLM Token", style="red")
    table.add_column("SmolLM ID", style="blue")
    table.add_column("BERT Fwd Prob", justify="right")
    table.add_column("SmolLM Prob", justify="right")
    
    # First, create a mapping of character positions to SmolLM tokens
    smol_token_map = {}
    current_pos = 0
    for i, smol_result in enumerate(smol_results):
        token_text = smol_tokenizer.decode([smol_result["token_id"]])
        token_len = len(token_text)
        smol_token_map[current_pos] = {
            "token": token_text,
            "token_id": smol_result["token_id"],
            "prob": smol_result["prob"]
        }
        current_pos += token_len
        logging.debug(f"SmolLM mapping: pos {current_pos-token_len} -> {token_text} (ID: {smol_result['token_id']})")
    
    # Process tokens
    for result in forward_results:
        bert_token = bert_tokenizer.decode([result["token_id"]])
        # Find corresponding SmolLM token based on character position
        smol_token_data = smol_token_map.get(result["original_start"])
        if smol_token_data is None:
            logging.warning(f"No SmolLM data found for position {result['original_start']}")
            continue
        
        smol_token_text = smol_token_data["token"]
        smol_token_id = smol_token_data["token_id"]
        smol_prob = smol_token_data["prob"]
        logging.debug(f"SmolLM token: {smol_token_text}, ID: {smol_token_id}, Prob: {smol_prob}")
        
        # Add row with only the most relevant information
        table.add_row(
            str(result["original_start"]),
            result["char"],
            bert_token,
            str(result["token_id"]),
            smol_token_text,
            str(smol_token_id),
            f"{result['forward']['prob']:.2f}",
            f"{smol_prob:.6f}" if smol_prob > 0 else "0.000000"
        )
    
    return table

def get_top_predictions(logits, tokenizer, top_n=3):
    """Get top n predictions from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_n)
    return [
        (tokenizer.decode([idx.item()]), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

def print_summary_statistics(example, tokenizer):
    """Print summary statistics for an example."""
    forward_results = example["bert_forward_results"]
    backward_results = example["bert_backward_results"]
    smol_results = example["smol_results"]
    
    # Calculate all metrics from demo6
    f_probs = [r["prob"] for r in forward_results]
    b_probs = [r["prob"] for r in backward_results]
    s_probs = [r["prob"] for r in smol_results]
    
    f_ratios = [get_clamped_ratio(r["prob"], r.get("neighbor_prob", 0.0)) for r in forward_results]
    b_ratios = [get_clamped_ratio(r["prob"], r.get("neighbor_prob", 0.0)) for r in backward_results]
    
    f_entropies = [r.get("entropy", 0.0) for r in forward_results]
    b_entropies = [r.get("entropy", 0.0) for r in backward_results]
    
    logging.info("\nSummary Statistics:")
    logging.info("Forward Analysis:")
    logging.info(f"  Probabilities: avg={np.mean(f_probs):.4f}, median={np.median(f_probs):.4f}, entropy={np.mean(f_entropies):.4f}")
    logging.info(f"  Ratios: avg={np.mean(f_ratios):.4f}, median={np.median(f_ratios):.4f}")
    
    logging.info("\nBackward Analysis:")
    logging.info(f"  Probabilities: avg={np.mean(b_probs):.4f}, median={np.median(b_probs):.4f}, entropy={np.mean(b_entropies):.4f}")
    logging.info(f"  Ratios: avg={np.mean(b_ratios):.4f}, median={np.median(b_ratios):.4f}")
    
    logging.info("\nSmolLM Analysis:")
    logging.info(f"  Probabilities: avg={np.mean(s_probs):.4f}, median={np.median(s_probs):.4f}")

def main():
    """Main visualization function."""
    logging.info("Starting visualization of analysis results...")
    
    # Load tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
    smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # Use consistent path handling
    script_dir = Path(__file__).parent
    results_file = script_dir / "analysis" / "analysis_results.json"
    
    logging.info(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create visualizations for each example
    for i, example in enumerate(results["examples"], 1):
        logging.info(f"\nAnalyzing Example {i}:")
        logging.info(f"Text: {example['text']}\n")
        
        # Create combined visualization
        html_output = visualize_combined_results(
            example["bert_forward_results"],
            example["bert_backward_results"],
            example["smol_results"],
            bert_tokenizer,
            example["text"]
        )
        
        # Save HTML output
        output_file = script_dir / "analysis" / f"example_{i}_analysis.html"
        with open(output_file, 'w') as f:
            f.write(html_output)
        logging.info(f"Saved visualization to {output_file}")
        
        # Print to console
        console = Console()
        console.print("\nToken-by-token Analysis:")
        create_console_table(
            example["bert_forward_results"],
            example["bert_backward_results"],
            example["smol_results"],
            bert_tokenizer,
            smol_tokenizer,
            console
        )
        
        # Print summary statistics
        print_summary_statistics(example, bert_tokenizer)

if __name__ == "__main__":
    main() 