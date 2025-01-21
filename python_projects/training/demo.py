import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch
from pathlib import Path
import logging
import re
import asyncio
import signal
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

async def handle_interrupt(loop):
    """Handle interrupt signal gracefully"""
    log.info("Received interrupt signal. Stopping gracefully...")
    loop.stop()

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

async def generate_text_async(model, tokenizer, prompt, max_length=50, temperature=0.7, num_samples=3):
    """Generate text using the model asynchronously"""
    print("\nTokenization breakdown:")
    print("-" * 60)
    print(f"{'Token ID':<10} {'Token':<20} {'Raw Token':<20} {'String'}")
    print("-" * 60)
    
    # Try encoding with original prompt
    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    
    # If no tokens, try with space prefix
    if len(encoded) == 0:
        print("No tokens produced, trying with space prefix...")
        prompt = ' ' + prompt
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
    
    # If still no tokens, try finding a similar token
    if len(encoded) == 0:
        print("Input produced no tokens, trying to find a similar token...")
        # Find a token that starts with the input
        for i in range(tokenizer.vocab_size):
            token = tokenizer.decode([i])
            if token.lower().startswith(prompt.lower()):
                encoded = [i]
                print(f"Using similar token: ID {i}: {repr(token)}")
                break
    
    # If still no tokens, show error
    if len(encoded) == 0:
        print("Error: Could not find any suitable token for the input!")
        return []
    
    input_ids = torch.tensor([encoded])
    
    # Show tokenization details
    print("\nFull tokenization:")
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    for token_id, token in zip(encoded, tokens):
        # Get the string representation
        token_str = tokenizer.convert_tokens_to_string([token])
        print(f"{token_id:<10} {token:<20} {repr(token):<20} {repr(token_str)}")
    print("-" * 60)
    print(f"Total tokens: {len(encoded)}\n")
    
    print(f'Generating {num_samples} samples with prompt: {repr(prompt)}')
    print(f'Parameters: max_length={max_length}, temperature={temperature}\n')
    
    samples = []
    for i in range(num_samples):
        print(f"Sample {i+1} generation details:")
        
        # Generate asynchronously
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            outputs = await loop.run_in_executor(
                None, 
                partial(
                    model.generate,
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(generated_text)
        
        # Show token-by-token breakdown of the generated sequence
        print("\nGenerated sequence tokens:")
        print("-" * 60)
        generated_tokens = tokenizer.convert_ids_to_tokens(outputs[0].tolist())
        for token_id, token in zip(outputs[0].tolist(), generated_tokens):
            # Get the string representation
            token_str = tokenizer.convert_tokens_to_string([token])
            print(f"{token_id:<10} {token:<20} {repr(token):<20} {repr(token_str)}")
        print("-" * 60)
        print(f"Generated text: {repr(generated_text)}\n")
    
    return samples

async def main():
    
    loop = asyncio.get_event_loop()
    # Set signal handler for interrupt using asyncio
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(handle_interrupt(loop)))

    
    model_path = Path("./tiny_gpt")
    
    if not model_path.exists():
        log.error(f"Model directory {model_path} not found! Please train the model first.")
        return
    
    log.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    log.info(f"Model and tokenizer loaded from {model_path}")
    
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
            
            # Show all vocabulary tokens
            log.info("\nAll vocabulary tokens:")
            for id in range(vocab_size):
                token = tokenizer.decode([id])
                log.info(f"  ID {id:4d}: {token}")
            continue
            
        if prompt == "/quit":
            break
        
        # Use await for the async generation function
        samples = await generate_text_async(
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

if __name__ == "__main__":
    asyncio.run(main()) 