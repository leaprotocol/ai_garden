import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from demo6_compute import compute_analysis
from demo6_visualize import visualize_combined_results
from rich.console import Console
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .token {{ font-weight: bold; }}
        .context {{ color: #666; }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

def run_demo_suite():
    """Run the complete demo suite with a simple test sentence."""
    console = Console()
    
    try:
        # Load models and tokenizers
        models = load_models_and_tokenizers()
        if not models:
            logging.error("Failed to load models")
            return
            
        # Test sentence
        test_sentence = "The quick brown fox jumps over the lazy dog."
        test_sentence = test_sentence.lower()
        
        # Compute analysis
        analysis_results = compute_analysis_with_logging(models, test_sentence)
        if not analysis_results:
            logging.error("Analysis failed")
            return
            
        # Visualize results
        visualize_results(analysis_results, models["bert_tokenizer"], test_sentence, console)
        
    except Exception as e:
        logging.error(f"Error in demo suite: {str(e)}")
        raise

def load_models_and_tokenizers():
    """Load and return models and tokenizers with error handling."""
    logging.info("Loading models and tokenizers...")
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        bert_model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-small")
        
        smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        smol_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        
        return {
            "bert_tokenizer": bert_tokenizer,
            "bert_model": bert_model,
            "smol_tokenizer": smol_tokenizer,
            "smol_model": smol_model
        }
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return None

def compute_analysis_with_logging(models, test_sentence):
    """Compute analysis with detailed logging."""
    logging.info("Computing analysis...")
    try:
        return compute_analysis(
            bert_model=models["bert_model"],
            bert_tokenizer=models["bert_tokenizer"],
            smol_model=models["smol_model"],
            smol_tokenizer=models["smol_tokenizer"],
            examples=[test_sentence]
        )
    except Exception as e:
        logging.error(f"Error in analysis computation: {str(e)}")
        return None

def visualize_results(analysis_results, tokenizer, test_sentence, console):
    """Handle visualization of results."""
    logging.info("Visualizing results...")
    
    # Extract results for the first (and only) example
    example_results = analysis_results["examples"][0]
    
    logging.debug(f"Text before alignment: '{test_sentence}'")
    # Align tokens with positions
    aligned_results, token_boundaries = align_tokens_with_positions(
        example_results["bert_forward_results"], 
        example_results["bert_backward_results"], 
        example_results["smol_results"], 
        tokenizer,
        test_sentence
    )
    
    # Create console visualization
    create_console_table(aligned_results, console)
    
    # Create HTML visualization
    html_output = visualize_combined_results(
        example_results["bert_forward_results"], 
        example_results["bert_backward_results"], 
        example_results["smol_results"], 
        tokenizer,
        test_sentence
    )
    
    # Save HTML output
    with open("demo7_analysis.html", "w") as f:
        f.write(html_output)
    logging.info("Saved HTML visualization to demo7_analysis.html")

def create_console_table(aligned_results, console):
    """Create console table with position information and token IDs."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Char Pos", style="cyan")
    table.add_column("Char", justify="center")
    table.add_column("BERT Token", style="green")
    table.add_column("BERT ID", justify="right")
    table.add_column("SmolLM Token", style="yellow")
    table.add_column("SmolLM ID", justify="right")
    table.add_column("BERT Fwd Prob", justify="right")
    table.add_column("SmolLM Prob", justify="right")

    logging.debug("--- Aligned Results for Console Table ---")
    logging.debug(aligned_results)

    for result in aligned_results:
        # Get all tokens at this position
        tokens = result['tokens']
        if not tokens:
            continue
            
        # Get first token (we'll show one row per character position)
        token = tokens[0]
        smol = token.get('smol', {})
        
        table.add_row(
            str(result['char_pos']),
            result['char'],
            token['token'],
            str(token['token_id']),
            smol.get('token', ""),
            str(smol.get('token_id', "")),
            f"{token['forward']['prob']:.2f}",
            f"{smol.get('prob', 0.0):.2f}"
        )
    
    console.print(table)

def visualize_combined_results(forward_results, backward_results, smol_results, tokenizer, original_text=None):
    """Create a table showing forward and backward analysis results."""
    logging.info("Starting HTML generation...")
    
    # Align tokens from different tokenizers
    aligned_results, token_boundaries = align_tokens(forward_results, backward_results, smol_results, tokenizer)
    
    # Use the original text if provided, otherwise reconstruct from tokens
    if original_text:
        logging.info("Using original text from JSON")
        cleaned_text = original_text
    else:
        logging.info("Decoding tokens for full text...")
        token_ids = [r['token_id'] for r in aligned_results]
        cleaned_text = tokenizer.decode(token_ids)
    
    # Create the sentence section
    logging.info(f"Creating sentence section with text length: {len(cleaned_text)}")
    sentence_section = f'<div class="sentence"><h3>Full Text:</h3><p>{cleaned_text}</p></div>\n'
    
    # Create the main analysis table
    logging.info("Creating main analysis table...")
    main_table = create_html_analysis_table(aligned_results, tokenizer)
    
    # Create the tokenization comparison table
    logging.info("Creating tokenization comparison table...")
    tokenization_table = create_tokenization_comparison_table(aligned_results, tokenizer)
    
    # Return the complete HTML using the template
    logging.info("Applying HTML template...")
    return HTML_TEMPLATE.format(content=sentence_section + main_table + tokenization_table)

def create_tokenization_comparison_table(aligned_results, tokenizer):
    """Create HTML table showing tokenization differences."""
    table = """
    <div class="tokenization-comparison">
        <h3>Tokenization Comparison</h3>
        <table>
            <thead>
                <tr>
                    <th>Char Pos</th>
                    <th>Char</th>
                    <th>BERT Token</th>
                    <th>SmolLM Token</th>
                    <th>BERT Fwd Prob</th>
                    <th>BERT Bwd Prob</th>
                    <th>SmolLM Prob</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in aligned_results:
        tokens = result['tokens']
        if not tokens:
            continue
            
        # Get first token (we'll show one row per character position)
        token = tokens[0]
        
        # Get BERT token
        bert_token = token['token']
        
        # Get SmolLM token, defaulting to bert_token if not available
        smol_token = token['smol'].get('token', bert_token) if token['smol'] else bert_token
        
        # Get probabilities
        bert_fwd_prob = token['forward']['prob']
        bert_bwd_prob = token['backward']['prob'] if token['backward'] else 0.0
        smol_prob = token['smol']['prob'] if token['smol'] else 0.0
        
        table += f"""
        <tr>
            <td>{result['char_pos']}</td>
            <td>{result['char']}</td>
            <td>{bert_token}</td>
            <td>{smol_token}</td>
            <td>{bert_fwd_prob:.2f}</td>
            <td>{bert_bwd_prob:.2f}</td>
            <td>{smol_prob:.2f}</td>
        </tr>
        """
    
    table += """
            </tbody>
        </table>
    </div>
    """
    return table

def create_html_analysis_table(aligned_results, tokenizer):
    """Create the main analysis table."""
    table = """
    <div class="analysis-table">
        <h3>Detailed Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Char Pos</th>
                    <th>Char</th>
                    <th>F.Prob</th>
                    <th>F.Neigh</th>
                    <th>F.Ent</th>
                    <th>F.Loss</th>
                    <th>Token</th>
                    <th>B.Prob</th>
                    <th>B.Ent</th>
                    <th>B.Loss</th>
                    <th>S.Prob</th>
                    <th>S.Ent</th>
                    <th>S.Loss</th>
                    <th>ID</th>
                    <th>Context</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in aligned_results:
        tokens = result['tokens']
        if not tokens:
            continue
            
        token = tokens[0]
        b_data = token['backward'] or {'prob': 0.0, 'entropy': 0.0, 'loss': 0.0}
        s_data = token['smol'] or {'prob': 0.0, 'entropy': 0.0, 'loss': 0.0}
        
        # Get colors for each metric
        f_prob_color = get_prob_color(token['forward']['prob'], False)
        b_prob_color = get_prob_color(b_data['prob'], True)
        s_prob_color = get_prob_color(s_data['prob'], False)
        
        f_ent_color = get_entropy_color(token['forward']['entropy'])
        b_ent_color = get_entropy_color(b_data['entropy'])
        s_ent_color = get_entropy_color(s_data['entropy'])
        
        f_loss_color = get_loss_color(token['forward']['loss'])
        b_loss_color = get_loss_color(b_data['loss'])
        s_loss_color = get_loss_color(s_data['loss'])
        
        table += f"""
        <tr>
            <td>{result['char_pos']}</td>
            <td>{result['char']}</td>
            <td style="color: {f_prob_color}">{token['forward']['prob']:.2f}</td>
            <td style="color: {f_prob_color}">{token['forward'].get('neighbor_prob', 0.0):.2f}</td>
            <td style="color: {f_ent_color}">{token['forward']['entropy']:.1f}</td>
            <td style="color: {f_loss_color}">{token['forward']['loss']:.1f}</td>
            <td class="token">{token['token']}</td>
            <td style="color: {b_prob_color}">{b_data['prob']:.2f}</td>
            <td style="color: {b_ent_color}">{b_data['entropy']:.1f}</td>
            <td style="color: {b_loss_color}">{b_data['loss']:.1f}</td>
            <td style="color: {s_prob_color}">{s_data['prob']:.2f}</td>
            <td style="color: {s_ent_color}">{s_data['entropy']:.1f}</td>
            <td style="color: {s_loss_color}">{s_data['loss']:.1f}</td>
            <td>{token['token_id']}</td>
            <td class="context">{get_context(tokens, tokenizer)}</td>
        </tr>
        """
    
    table += """
            </tbody>
        </table>
    </div>
    """
    return table

def align_tokens(forward_results, backward_results, smol_results, tokenizer):
    """Align tokens from different tokenizers for comparison."""
    aligned_results = []
    token_boundaries = []
    
    for i, (f, b, s) in enumerate(zip(forward_results, backward_results, smol_results)):
        # Get BERT token
        bert_token = tokenizer.decode([f['token_id']]).strip()
        
        # Create token boundary information
        boundary = {
            'token_id': f['token_id'],
            'token': bert_token,
            'original_start': i,
            'original_end': i + 1,
            'suggested_start': i,
            'suggested_end': i + 1,
            'forward': {
                'prob': f['prob'],
                'neighbor_prob': f.get('neighbor_prob', 0.0),
                'entropy': f['entropy'],
                'loss': f['loss']
            },
            'backward': {
                'prob': b['prob'] if b else 0.0,
                'entropy': b['entropy'] if b else 0.0,
                'loss': b['loss'] if b else 0.0
            } if b else None,
            'smol': {
                'token': tokenizer.decode([s['token_id']]) if s else "",
                'prob': s['prob'] if s else 0.0,
                'entropy': s['entropy'] if s else 0.0,
                'loss': s['loss'] if s else 0.0
            } if s else None
        }
        token_boundaries.append(boundary)
        
        # Create aligned result
        aligned_results.append({
            'char_pos': i,
            'char': bert_token[0] if bert_token else "",
            'tokens': [boundary]
        })
    
    return aligned_results, token_boundaries

def get_context(tokens, tokenizer, window_size=3):
    """Get context around a token."""
    if not tokens:
        return ""
        
    # Get the first token's position
    pos = tokens[0]['suggested_start']
    start = max(0, pos - window_size)
    end = min(len(tokens), pos + window_size + 1)
    
    context_tokens = []
    for i in range(start, end):
        if i == pos:
            context_tokens.append(f"[{tokens[i]['token']}]")
        else:
            context_tokens.append(tokens[i]['token'])
    
    return " ".join(context_tokens)

def get_prob_color(prob: float, is_backward: bool) -> str:
    """Get color for probability value."""
    if is_backward:
        # Backward probabilities use red scale
        red = min(255, int(255 * (1 - prob)))
        return f"rgb({red}, 0, 0)"
    else:
        # Forward probabilities use green scale
        green = min(255, int(255 * prob))
        return f"rgb(0, {green}, 0)"

def get_entropy_color(entropy: float) -> str:
    """Get color for entropy value."""
    # Blue scale - higher entropy = darker blue
    blue = min(255, int(255 * (entropy / 10)))  # Assuming max entropy of 10
    return f"rgb(0, 0, {blue})"

def get_loss_color(loss: float) -> str:
    """Get color for loss value."""
    # Purple scale - higher loss = darker purple
    purple = min(255, int(255 * (loss / 10)))  # Assuming max loss of 10
    return f"rgb({purple}, 0, {purple})"

def align_tokens_with_positions(forward_results, backward_results, smol_results, tokenizer, original_text):
    """Align tokens with unified position tracking."""
    aligned_results = []
    current_pos = 0

    logging.debug(f"Original text: {original_text}")
    logging.debug(f"Original text length: {len(original_text)}")

    # Create a mapping of character positions to token boundaries
    token_boundaries = []
    for i, (f, b, s) in enumerate(zip(forward_results, backward_results, smol_results)):
        # Get token text
        token_text = tokenizer.decode([f['token_id']]).strip()
        logging.debug(f"Processing token {i}: '{token_text}' (length: {len(token_text)})")
        logging.debug(f"  Searching for token: '{token_text}' from position: {current_pos}")

        start = original_text.find(token_text, current_pos)
        end = start + len(token_text) if start != -1 else current_pos + len(token_text)

        if start != -1:
            # Token found, advance current_pos to the end of the token
            current_pos = end
        else:
            # Token not found, advance current_pos by 1 to continue search
            current_pos += 1
            logging.warning(f"Token '{token_text}' not found in text, advancing current_pos by 1")

        # Create token boundary information
        boundary = {
            'token_id': f['token_id'],
            'token': token_text,
            'original_start': start,
            'original_end': end,
            'suggested_start': i,
            'suggested_end': i + 1,
            'forward': {
                'prob': f['prob'],
                'neighbor_prob': f.get('neighbor_prob', 0.0),
                'entropy': f['entropy'],
                'loss': f['loss']
            },
            'backward': {
                'prob': b['prob'] if b else 0.0,
                'entropy': b['entropy'] if b else 0.0,
                'loss': b['loss'] if b else 0.0
            } if b else None,
            'smol': {
                'token': tokenizer.decode([s['token_id']]) if s else "",
                'prob': s['prob'] if s else 0.0,
                'entropy': s['entropy'] if s else 0.0,
                'loss': s['loss'] if s else 0.0
            } if s else None
        }

        token_boundaries.append(boundary)
        logging.debug(f"  Token: '{token_text}', start: {start}, end: {end}, current_pos: {current_pos}")

    # Create aligned results by iterating through characters
    aligned_results = []
    for char_pos in range(len(original_text)):
        # Find all tokens that start or end at this position
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

if __name__ == "__main__":
    run_demo_suite() 


