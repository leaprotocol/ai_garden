import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch
from pathlib import Path
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def show_tokenization(tokenizer, text):
    """Show how the text is tokenized."""
    # Get the tokens and their IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Get the string representation of each token
    token_strings = [tokenizer.decode([token_id]) for token_id in tokens]
    
    print("\nTokenization breakdown:")
    print("-" * 60)
    print(f"{'Token ID':<10} {'Token':<20} {'Visualization'}")
    print("-" * 60)
    for token_id, token_str in zip(tokens, token_strings):
        # Show raw bytes for better visualization
        bytes_viz = ' '.join(f'{b:02x}' for b in token_str.encode('utf-8'))
        print(f"{token_id:<10} {token_str:<20} {bytes_viz}")
    print("-" * 60)
    print(f"Total tokens: {len(tokens)}")

def clean_text(text):
    """Clean up the generated text."""
    # Remove the Ġ characters but preserve word boundaries
    text = re.sub('Ġ', ' ', text)
    # Fix extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def format_token_text(token):
    """Format a token for display, preserving word boundaries."""
    # Remove Ġ but add a visible separator
    if token.startswith('Ġ'):
        return ' ' + token.replace('Ġ', '')
    return token

def show_token_details(model, tokenizer, input_ids, token_id, position):
    """Show detailed information about a generated token."""
    with torch.no_grad():
        # Get model's raw logits for this position
        outputs = model(input_ids[:, :position+1])
        logits = outputs.logits[:, -1, :]  # Get logits for the last position
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get top 5 most likely next tokens
        top_probs, top_indices = torch.topk(probs[0], k=5)
        
        # Print token details
        token_text = tokenizer.decode([token_id])
        log_prob = log_probs[0, token_id].item()
        prob = probs[0, token_id].item()
        
        print(f"\nToken {position}:")
        print(f"  Text: '{token_text}'")
        print(f"  ID: {token_id}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Log Probability: {log_prob:.4f}")
        
        print("\nTop 5 most likely tokens at this position:")
        for i, (p, idx) in enumerate(zip(top_probs, top_indices), 1):
            token = tokenizer.decode([idx])
            print(f"  {i}. '{token}' (ID: {idx}) - Prob: {p:.4f}")

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, num_samples=3):
    """Generate multiple samples of text from a prompt with token-level details."""
    # Show tokenization of the prompt
    show_tokenization(tokenizer, prompt)
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    
    # Generate multiple samples
    log.info(f"\nGenerating {num_samples} samples with prompt: '{prompt}'")
    log.info(f"Parameters: max_length={max_length}, temperature={temperature}")
    
    samples = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Generate one token at a time to show details
            input_ids = inputs["input_ids"].clone()
            attention_mask = inputs["attention_mask"].clone()
            
            generated_tokens = []
            current_length = input_ids.shape[1]
            
            if sample_idx == 0:  # Only show detailed generation for first sample
                print(f"\nSample {sample_idx + 1} generation details:")
                
            while current_length < max_length:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                if sample_idx == 0:  # Show details only for first sample
                    show_token_details(model, tokenizer, input_ids, next_token.item(), current_length)
                
                # Append the new token
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                generated_tokens.append(next_token.item())
                current_length += 1
                
                # Stop if we generate an end of text token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode the generated sequence
            generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Create a version with visible token boundaries for the first sample
            if sample_idx == 0:
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                # Show raw tokens first
                print("\nRaw tokens:")
                print(" ".join(tokens))
                
                # Show formatted tokens with boundaries
                print("\nGenerated text with token boundaries:")
                formatted_tokens = [format_token_text(t) for t in tokens]
                print("[" + "][".join(formatted_tokens) + "]")
                
                # Show clean text with token boundaries preserved
                clean_tokens = [t.replace('Ġ', '') for t in tokens]
                text_with_boundaries = ""
                for i, token in enumerate(clean_tokens):
                    if i > 0 and tokens[i].startswith('Ġ'):
                        text_with_boundaries += " " + token
                    else:
                        text_with_boundaries += token
                print("\nClean text with natural spacing:")
                print(text_with_boundaries)
            
            samples.append(clean_text(generated_text))
    
    return samples

def main():
    model_path = Path("./tiny_gpt")
    
    if not model_path.exists():
        log.error(f"Model directory {model_path} not found! Please train the model first.")
        return
    
    log.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    
    log.info("\nModel loaded! Available commands:")
    log.info("  /help    - Show this help")
    log.info("  /params  - Change generation parameters")
    log.info("  /tokens  - Show vocabulary tokens")
    log.info("  /quit    - Exit the demo")
    log.info("Enter prompts to generate text...")
    
    # Default parameters
    params = {
        "max_length": 50,
        "temperature": 0.7,
        "num_samples": 3
    }
    
    try:
        while True:
            prompt = input("\nEnter a prompt: ").strip()
            if not prompt:
                continue
                
            if prompt == "/help":
                log.info("\nAvailable commands:")
                log.info("  /help    - Show this help")
                log.info("  /params  - Change generation parameters")
                log.info("  /tokens  - Show vocabulary tokens")
                log.info("  /quit    - Exit the demo")
                continue
                
            if prompt == "/params":
                try:
                    params["max_length"] = int(input("Max length (current: {}): ".format(params["max_length"])) or params["max_length"])
                    params["temperature"] = float(input("Temperature (current: {}): ".format(params["temperature"])) or params["temperature"])
                    params["num_samples"] = int(input("Number of samples (current: {}): ".format(params["num_samples"])) or params["num_samples"])
                    log.info("\nParameters updated!")
                except ValueError as e:
                    log.error("Invalid input. Using previous values.")
                continue
                
            if prompt == "/tokens":
                # Show some vocabulary statistics
                vocab_size = tokenizer.vocab_size
                log.info(f"\nVocabulary size: {vocab_size}")
                log.info("\nSpecial tokens:")
                for token, id in tokenizer.special_tokens_map.items():
                    log.info(f"  {token}: {id} (ID: {tokenizer.convert_tokens_to_ids(id)})")
                
                # Show some example tokens
                log.info("\nSample vocabulary tokens:")
                sample_ids = torch.randint(0, vocab_size, (10,))
                for id in sample_ids:
                    token = tokenizer.decode([id.item()])
                    log.info(f"  ID {id:4d}: {token}")
                continue
                
            if prompt == "/quit":
                break
                
            samples = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=params["max_length"],
                temperature=params["temperature"],
                num_samples=params["num_samples"]
            )
            
            print("\nGenerated samples:")
            for i, sample in enumerate(samples, 1):
                print("\nSample", i)
                print("-" * 40)
                print(sample)
                print("-" * 40)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main() 