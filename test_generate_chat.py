import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import List, Tuple, Optional
from rich import print
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Rich Console
console = Console()


def load_model(model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_top_n_tokens(tokenizer, logits: torch.Tensor, n: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top N tokens and their probabilities from logits.
    
    Args:
        tokenizer: The tokenizer to decode token IDs
        logits: The model's output logits for next token prediction
        n: Number of top tokens to return
    
    Returns:
        List of tuples containing (token_text, probability)
    """
    probabilities = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=n)
    
    return [(tokenizer.decode(idx.item()).strip(), prob.item()) 
            for idx, prob in zip(top_indices, top_probs)]


def calculate_entropy(probabilities: torch.Tensor) -> float:
    """
    Calculate the entropy of the probability distribution.
    
    Args:
        probabilities: A tensor of probabilities.
    
    Returns:
        Entropy value.
    """
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
    return entropy.item()


def generate_tokens_with_probabilities(
    tokenizer, 
    model, 
    prompt: str, 
    num_tokens: int = 5, 
    top_n: int = 5
):
    """
    Generate tokens one by one, showing top N probabilities at each step.
    
    Args:
        tokenizer: The tokenizer associated with the model
        model: The language model
        prompt: Input text prompt
        num_tokens: Number of tokens to generate
        top_n: Number of top probability tokens to show at each step
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_tokens = []
    
    # Store data for visualization
    entropies = []
    step_data = []
    
    console.print(f"\n[bold blue]Starting generation from prompt:[/bold blue] {prompt}\n")
    
    for step in range(num_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)
            
            # Get top N tokens and their probabilities
            top_tokens = get_top_n_tokens(tokenizer, next_token_logits, top_n)
            entropy = calculate_entropy(probabilities)
            entropies.append(entropy)
            
            # Store step data for visualization
            step_dict = {'step': step + 1}
            for rank, (token, prob) in enumerate(top_tokens):
                step_dict[f'token_{rank+1}'] = token
                step_dict[f'prob_{rank+1}'] = prob
            step_data.append(step_dict)
            
            # Create and populate the probability table
            table = Table(title=f"Step {step + 1}: Top {top_n} Next Token Probabilities")
            table.add_column("Rank", justify="right", style="cyan")
            table.add_column("Token", style="magenta")
            table.add_column("Probability", justify="right", style="green")
            
            # Add rows to the table
            for rank, (token, prob) in enumerate(top_tokens, 1):
                if rank == 1:  # Highlight the selected token
                    table.add_row(f"#{rank}", f"[bold green]{token}[/bold green]", f"[bold green]{prob:.4f}[/bold green]")
                else:
                    table.add_row(f"#{rank}", f"'{token}'", f"{prob:.4f}")
            
            console.print(table)
            console.print(f"[bold yellow]Entropy:[/bold yellow] {entropy:.4f}\n")
            
            # Select the top token and add it to the sequence
            selected_token_id = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0)
            input_ids = torch.cat([input_ids, selected_token_id.unsqueeze(0)], dim=-1)
            generated_tokens.append(tokenizer.decode(selected_token_id))
    
    # Show final generated sequence
    console.print("\n[bold green]Final generated sequence:[/bold green]")
    console.print(f"{prompt}{''.join(generated_tokens)}")
    
    # Create visualizations
    plt.figure(figsize=(12, 4))
    
    # Plot entropy over steps
    plt.subplot(121)
    plt.plot(range(1, num_tokens + 1), entropies, marker='o')
    plt.title('Entropy Over Generation Steps')
    plt.xlabel('Generation Step')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    # Create probability heatmap
    plt.subplot(122)
    df = pd.DataFrame(step_data)
    prob_columns = [col for col in df.columns if col.startswith('prob_')]
    token_columns = [col for col in df.columns if col.startswith('token_')]
    
    # Create heatmap data
    heatmap_data = df[prob_columns].values
    token_labels = df[token_columns].iloc[0].values
    
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt=".3f", 
                cmap='Blues',
                xticklabels=token_labels,
                yticklabels=range(1, num_tokens + 1))
    plt.title('Token Probabilities Heatmap')
    plt.xlabel('Tokens')
    plt.ylabel('Generation Step')
    
    plt.tight_layout()
    plt.show()


def generate_tokens_with_attention(
    tokenizer, 
    model, 
    prompt: str, 
    num_tokens: int = 5, 
    top_n: int = 5
):
    """
    Generate tokens and visualize attention weights if available.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_tokens = []
    
    console.print(f"\n[bold blue]Starting generation from prompt:[/bold blue] {prompt}\n")
    
    for step in range(num_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            next_token_logits = outputs.logits[0, -1, :]
            attentions = outputs.attentions  # Tuple of attention tensors
            
            # Proceed with token generation and visualization...
            # (Similar to previous functions, plus handling attentions)


def predict_token_probability(
    tokenizer, 
    model, 
    sentence: str, 
    next_token: str
) -> float:
    """
    Predict the probability of a specific next token given a sentence.
    
    Args:
        tokenizer: The tokenizer associated with the model.
        model: The pre-trained language model.
        sentence: The input sentence.
        next_token: The token whose probability we want to predict.
    
    Returns:
        Probability of the next token.
    """
    # Encode the input sentence
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    
    # Get the model's output logits for the next token
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
    
    # Calculate probabilities
    probabilities = F.softmax(next_token_logits, dim=-1)
    
    # Encode the next token to get its ID
    next_token_id = tokenizer.encode(next_token, add_special_tokens=False)
    
    if len(next_token_id) != 1:
        raise ValueError("The next token should be a single token as per the tokenizer.")
    
    # Get the probability of the specified next token
    token_probability = probabilities[next_token_id[0]].item()
    
    return token_probability


def load_model(model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def visualize_sentence_probabilities(
    tokenizer, 
    model, 
    sentence: str, 
    next_token: str
):
    """
    Visualize the probability of a specific next token after each word in a sentence.
    
    Args:
        tokenizer: The tokenizer associated with the model.
        model: The pre-trained language model.
        sentence: The input sentence.
        next_token: The token whose probability we want to visualize.
    """
    # Encode the next token to get its ID
    next_token_id = tokenizer.encode(next_token, add_special_tokens=False)
    
    if len(next_token_id) != 1:
        raise ValueError("The next token should be a single token as per the tokenizer.")
    
    # Tokenize the sentence
    words = sentence.split()
    
    # Iterate over each word in the sentence
    for i in range(len(words)):
        # Create a partial sentence up to the current word
        partial_sentence = " ".join(words[:i+1])
        
        # Encode the partial sentence
        input_ids = tokenizer.encode(partial_sentence, return_tensors='pt')
        
        # Get the model's output logits for the next token
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
        
        # Calculate probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        # Get the probability of the specified next token
        token_probability = probabilities[next_token_id[0]].item()
        
        # Print the probability after the current word
        print(f"Probability of '{next_token}' after '{partial_sentence}': {token_probability:.4f}")

def normalize_tokens(tokenizer, text: str) -> List[List[int]]:
    """
    Get all possible tokenizations of a text.
    Returns a list of token ID sequences that represent the same text.
    """
    variants = [
        text,
        text.strip(),
        f" {text.strip()}",
        f"{text.strip()} "
    ]
    
    tokenizations = []
    seen = set()
    
    for variant in variants:
        token_ids = tuple(tokenizer.encode(variant, add_special_tokens=False))
        if token_ids not in seen:
            seen.add(token_ids)
            tokenizations.append(list(token_ids))
    
    return tokenizations

def get_top_n_token_pairs(
    tokenizer, 
    model, 
    input_ids: torch.Tensor, 
    actual_next_words: str,
    k: int = 5,
    beam_width: int = 15  # Increase beam width to catch more variations
) -> Tuple[List[Tuple[str, List[int], float]], float, List[int]]:
    """
    Get the top N most probable pairs of consecutive tokens with improved beam search.
    """
    pairs = []
    actual_pair_prob = None
    possible_tokenizations = normalize_tokens(tokenizer, actual_next_words)
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        first_token_probs = F.softmax(next_token_logits, dim=-1)
    
    # Increase beam width to catch more tokenization variations
    top_first_probs, top_first_indices = torch.topk(first_token_probs, k=beam_width)
    
    # Track all variations of the first token
    first_token_variations = {}
    for first_idx, first_prob in zip(top_first_indices, top_first_probs):
        decoded = tokenizer.decode([first_idx.item()]).strip()
        if decoded not in first_token_variations:
            first_token_variations[decoded] = []
        first_token_variations[decoded].append((first_idx.item(), first_prob.item()))
    
    # Process each first token variation
    for decoded_first, variations in first_token_variations.items():
        for first_idx, first_prob in variations:
            new_input_ids = torch.cat([input_ids, torch.tensor([[first_idx]], device=input_ids.device)], dim=-1)
            
            with torch.no_grad():
                outputs = model(new_input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                second_token_probs = F.softmax(next_token_logits, dim=-1)
            
            top_second_probs, top_second_indices = torch.topk(second_token_probs, k)  # Increased from 5
            
            for second_idx, second_prob in zip(top_second_indices, top_second_probs):
                token_ids = [first_idx, second_idx.item()]
                pair_text = tokenizer.decode(token_ids).strip()
                joint_prob = first_prob * second_prob.item()
                
                # Only add if it's a new unique pair text
                if not any(p[0] == pair_text for p in pairs):
                    pairs.append((pair_text, token_ids, joint_prob))
                
                # Check all possible tokenizations
                if any(pair_text == tokenizer.decode(t[:2]).strip() for t in possible_tokenizations):
                    if actual_pair_prob is None or joint_prob > actual_pair_prob:
                        actual_pair_prob = joint_prob
    
    # Sort by probability and deduplicate based on text
    pairs.sort(key=lambda x: x[2], reverse=True)
    unique_pairs = []
    seen = set()
    for pair in pairs:
        if pair[0] not in seen:
            seen.add(pair[0])
            unique_pairs.append(pair)
    
    return unique_pairs[:k], actual_pair_prob, possible_tokenizations[0]

def print_token_info(tokenizer, text: str):
    """
    Print detailed tokenization information for a given text.
    """
    # Get token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Get individual tokens
    tokens = tokenizer.tokenize(text)
    
    print(f"\n[bold cyan]Token analysis for '{text}':[/bold cyan]")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")
    
    # Try different variants
    variants = [text, text.lower(), text.upper(), f" {text}", f"{text} "]
    for variant in variants:
        variant_ids = tokenizer.encode(variant, add_special_tokens=False)
        if variant_ids != token_ids:
            print(f"Different tokenization for '{variant}': {variant_ids}")

def visualize_actual_next_word_probabilities(
    tokenizer, 
    model, 
    sentence: str
):
    """
    Visualize the probability of each actual next word pair in a sentence.
    """
    words = sentence.split()
    
    for i in range(len(words) - 2):
        partial_sentence = " ".join(words[:i+1])
        actual_next_words = " ".join(words[i+1:i+3])
        
        # Print token analysis for the actual next words
        print_token_info(tokenizer, actual_next_words)
        
        input_ids = tokenizer.encode(partial_sentence, return_tensors='pt')
        
        # Get predictions and actual token information
        top_pairs, actual_pair_prob, actual_token_ids = get_top_n_token_pairs(
            tokenizer, model, input_ids, actual_next_words
        )
        
        # Print context
        print(f"\n[bold blue]After: '{partial_sentence}'[/bold blue]")
        print(f"[bold green]Actual next words: '{actual_next_words}' (tokens: {actual_token_ids})[/bold green]")
        
        # Create table
        table = Table(title=f"Next Token Pair Probabilities")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Token Pair", style="magenta")
        table.add_column("Token IDs", style="yellow")
        table.add_column("Probability", justify="right", style="green")
        
        # Add top pairs to table
        actual_in_top = False
        for rank, (pair_text, token_ids, prob) in enumerate(top_pairs, 1):
            if pair_text.strip() == actual_next_words.strip():
                table.add_row(
                    f"#{rank}",
                    f"[bold yellow]'{pair_text}'[/bold yellow]",
                    f"[bold yellow]{token_ids}[/bold yellow]",
                    f"[bold yellow]{prob:.4f}[/bold yellow]"
                )
                actual_in_top = True
            else:
                table.add_row(f"#{rank}", f"'{pair_text}'", f"{token_ids}", f"{prob:.4f}")
        
        # Always add actual pair if it has a probability, even if it's 0
        if not actual_in_top:
            table.add_row("---", "---", "---", "---")
            prob_str = f"{actual_pair_prob:.4f}" if actual_pair_prob is not None else "0.0000"
            table.add_row(
                "N/A",
                f"[bold red]'{actual_next_words}'[/bold red]",
                f"[bold red]{actual_token_ids[:2]}[/bold red]",
                f"[bold red]{prob_str}[/bold red]"
            )
        
        console.print(table)
        console.print()

def get_token_embeddings(model):
    """
    Extract token embeddings from the model's embedding layer.
    """
    # Most transformer models store embeddings in either wte or embed_tokens
    if hasattr(model, 'transformer'):
        return model.transformer.wte.weight
    elif hasattr(model, 'model'):
        return model.model.embed_tokens.weight
    else:
        raise AttributeError("Could not find embedding layer")

def get_color_for_similarity(sim: float) -> str:
    """
    Returns a color based on the similarity value using a red -> yellow -> green gradient.
    -1.0 -> red
     0.0 -> desaturated yellow 
    +1.0 -> green
    """
    # Convert similarity from [-1,1] to [0,1] range
    normalized = (sim + 1) / 2
    
    if normalized < 0.5:
        # Red to yellow gradient
        # As we go from 0 to 0.5, increase green component
        red = 255
        green = int(255 * (normalized * 2))
        return f"[rgb({red},{green},0)]"
    else:
        # Yellow to green gradient
        # As we go from 0.5 to 1, decrease red component
        red = int(255 * (2 - normalized * 2))
        green = 255
        return f"[rgb({red},{green},0)]"

def analyze_token_similarity(tokenizer, model, tokens_of_interest):
    """
    Analyze similarity between different tokenizations of the same word/phrase.
    """
    embeddings = get_token_embeddings(model)
    
    for text in tokens_of_interest:
        print(f"\n[bold cyan]Token similarity analysis for '{text}':[/bold cyan]")
        
        # Get all possible tokenizations
        variants = [
            text,
            text.lower(),
            text.upper(),
            f" {text}",
            f"{text} "
        ]
        
        # Collect all unique token IDs and their decoded forms
        all_tokens = {}
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            for token_id, token in zip(token_ids, tokens):
                if token_id not in all_tokens:
                    all_tokens[token_id] = token
        
        # Create similarity matrix
        token_ids = list(all_tokens.keys())
        token_embeds = embeddings[token_ids].detach().cpu().numpy()
        sim_matrix = cosine_similarity(token_embeds)
        
        # Create and display similarity table
        table = Table(
            title=f"Token Similarity Matrix for '{text}'",
            show_header=True,
            header_style="bold white"
        )
        table.add_column("Token ID", justify="right", style="cyan")
        table.add_column("Token", style="magenta")
        
        for other_id in token_ids:
            table.add_column(str(other_id), justify="right")
        
        for i, token_id in enumerate(token_ids):
            row = [str(token_id), all_tokens[token_id]]
            
            # Add colored similarity values
            for sim in sim_matrix[i]:
                color = get_color_for_similarity(sim)
                row.append(f"{color}{sim:.3f}")
            
            table.add_row(*row)
        
        console.print(table)
        
        # Find most similar tokens in vocabulary
        print("\nMost similar tokens:")
        for token_id in token_ids:
            token_embed = embeddings[token_id].unsqueeze(0)
            sims = F.cosine_similarity(token_embed, embeddings)
            top_sims, top_ids = torch.topk(sims, k=5)
            
            print(f"\n{token_id} ('{all_tokens[token_id]}'):")
            for sim, tid in zip(top_sims, top_ids):
                color = get_color_for_similarity(sim.item())
                print(f"  {tokenizer.decode([tid])} ({tid}): {color}{sim:.3f}")

def get_top_n_token_sequences(
    tokenizer, 
    model, 
    input_ids: torch.Tensor,
    max_tokens: int = 10,
    top_n_best: int = 5,
    top_k_beam_width: int = 5,
    temperature: float = 1.0,
    end_tokens: Optional[List[str]] = None
) -> Tuple[List[Tuple[str, List[int], float, int]], dict]:
    """
    Get the top K most probable sequences up to max_tokens length.
    """
    stats = {
        'total_sequences_explored': 0,
        'sequences_per_depth': {},
        'pruned_sequences': 0,
        'completed_sequences': 0,
        'early_stops': {
            'special_token': 0,
            'punctuation': 0
        }
    }
    
    # Initialize sequences
    current_sequences = [([], 1.0)]  # Start with empty sequence
    candidate_sequences = []  # Store all candidates with their probabilities
    
    # Initialize end token IDs set
    end_token_ids = set()
    special_token_ids = set()
    
    # Add special token IDs if they exist
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        end_token_ids.add(tokenizer.eos_token_id)
        special_token_ids.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        end_token_ids.add(tokenizer.sep_token_id)
        special_token_ids.add(tokenizer.sep_token_id)
    
    # Add custom end tokens if provided, otherwise use defaults
    custom_end_tokens = end_tokens if end_tokens is not None else []
    
    # Add encoded punctuation tokens
    for token in custom_end_tokens:
        if isinstance(token, str):  # Only encode string tokens
            try:
                ids = tokenizer.encode(token, add_special_tokens=False)
                if ids:  # Some tokens might not encode to anything
                    end_token_ids.add(ids[-1])  # Use last token if multi-token
            except Exception as e:
                print(f"Warning: Could not encode token '{token}': {e}")
    
    print(f"Using end token IDs: {end_token_ids}")  # Debug info

    for step in range(max_tokens):
        new_sequences = []
        depth_sequences = 0
        
        for seq_tokens, seq_prob in current_sequences:
            # Skip if sequence already ended
            if seq_tokens and seq_tokens[-1] in end_token_ids:
                continue
                
            # Get model predictions
            current_input = torch.cat([
                input_ids,
                torch.tensor([t for t in seq_tokens]).unsqueeze(0)
            ], dim=1) if seq_tokens else input_ids
            
            with torch.no_grad():
                outputs = model(current_input)
                next_token_logits = outputs.logits[0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Get top K next tokens
            top_probs, top_indices = torch.topk(next_token_probs, k=top_k_beam_width)
            
            # Process each possibility
            for token_id, prob in zip(top_indices, top_probs):
                token_id = token_id.item()
                new_seq = seq_tokens + [token_id]
                new_prob = seq_prob * prob.item()
                depth_sequences += 1
                stats['total_sequences_explored'] += 1
                
                # Check if sequence ends
                is_complete = token_id in end_token_ids
                if is_complete:
                    if token_id in special_token_ids:
                        stats['early_stops']['special_token'] += 1
                    else:
                        stats['early_stops']['punctuation'] += 1
                
                # Add to candidates if:
                # 1. It's complete (ended with end token)
                # 2. It's at max length
                # 3. It's a shorter sequence with good probability
                if is_complete or len(new_seq) == max_tokens or new_prob > 0.1:  # Threshold for keeping shorter sequences
                    candidate_sequences.append((new_seq, new_prob, len(new_seq), is_complete))
                    stats['completed_sequences'] += 1
                
                # Continue sequence if not complete
                if not is_complete:
                    new_sequences.append((new_seq, new_prob))
        
        stats['sequences_per_depth'][step + 1] = depth_sequences
        
        # Keep top K sequences for next iteration
        new_sequences = sorted(new_sequences, key=lambda x: x[1], reverse=True)
        stats['pruned_sequences'] += max(0, len(new_sequences) - top_n_best)
        current_sequences = new_sequences[:top_n_best]
    
    # Sort all candidates by probability
    candidate_sequences = sorted(candidate_sequences, key=lambda x: x[1], reverse=True)
    
    # Convert top K to final format
    results = []
    seen_texts = set()
    
    for token_ids, prob, length, is_complete in candidate_sequences:
        text = tokenizer.decode(token_ids).strip()
        if text not in seen_texts:
            seen_texts.add(text)
            results.append((text, token_ids, prob, length, is_complete))
            if len(results) >= top_n_best:
                break
    
    return results, stats

def visualize_next_token_sequences(
    tokenizer,
    model,
    sentence: str,
    max_tokens: int = 10,
    top_n_best: int = 5,
    top_k_beam_width: int = 5,
    temperature: float = 1.0,
    end_tokens: Optional[List[str]] = None
):
    """
    Visualize the highest probability sequences and search statistics.
    """
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    
    # Get predictions and stats
    sequences, stats = get_top_n_token_sequences(
        tokenizer, model, input_ids, 
        max_tokens=max_tokens, 
        top_n_best=top_n_best,
        top_k_beam_width=top_k_beam_width,
        temperature=temperature,
        end_tokens=end_tokens
    )
    
    # Create sequences table
    table = Table(title=f"Top {top_n_best} Sequences (up to {max_tokens} tokens)")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Length", justify="right", style="yellow")
    table.add_column("Complete", justify="center", style="green")
    table.add_column("Sequence", style="magenta")
    table.add_column("Token IDs", style="yellow")
    table.add_column("Probability", justify="right", style="green")
    
    # Add sequences to table
    for rank, (text, token_ids, prob, length, is_complete) in enumerate(sequences, 1):
        table.add_row(
            f"#{rank}",
            str(length),
            "✓" if is_complete else "",
            f"'{text}'",
            str(token_ids),
            f"{prob:.6f}"
        )
    
    print(f"\n[bold blue]After: '{sentence}'[/bold blue]")
    console.print(table)
    
    # Create stats table
    stats_table = Table(title="Beam Search Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="green")
    
    stats_table.add_row("Total Sequences Explored", str(stats['total_sequences_explored']))
    stats_table.add_row("Completed Sequences", str(stats['completed_sequences']))
    stats_table.add_row("Early Stops (Special Tokens)", str(stats['early_stops']['special_token']))
    stats_table.add_row("Early Stops (Punctuation)", str(stats['early_stops']['punctuation']))
    stats_table.add_row("Pruned Sequences", str(stats['pruned_sequences']))
    
    for depth, count in stats['sequences_per_depth'].items():
        stats_table.add_row(f"Sequences at Depth {depth}", str(count))
    
    console.print("\n")
    console.print(stats_table)

def main():
    import time
    
    start_time = time.time()
    
    print("\n[bold]Loading model...[/bold]")
    tokenizer, model = load_model("HuggingFaceTB/SmolLM2-360M-Instruct")
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # First analyze token similarities
    tokens_of_interest = [
        "want",
        "to",
        "understand",
        "I want",
        "want to"
    ]
    
    print("\n[bold]Analyzing token similarities:[/bold]")
    # analyze_token_similarity(tokenizer, model, tokens_of_interest)
    
    print("\n[bold]Now running the sequence prediction analysis:[/bold]")
    sentence = "System: You are a helpful assistant. Assistant: Hello. User: Exit now. Assistant: Sure, I'll do that."
    
    sequence_start = time.time()
    visualize_next_token_sequences(
        tokenizer, model, sentence, 
        max_tokens=10, 
        top_n_best=10, 
        top_k_beam_width=5, 
        temperature=1.0
    )
    sequence_time = time.time() - sequence_start
    
    print(f"\nSequence prediction completed in {sequence_time:.2f} seconds")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

