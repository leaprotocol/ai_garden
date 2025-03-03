import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import code  # For starting the interactive REPL
from typing import List, Union, Optional

# 1. Load tokenizer and model globally (outside any function or conditional block)
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
print(f"Loading tokenizer for: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    low_cpu_mem_usage=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
print(f"Model loaded on device: {device}")

def generate_text_from_weighted_embeddings(
    prompts: List[str], 
    weights: Optional[List[float]] = None,
    max_length: int = 50
) -> Optional[str]:
    """
    Generates text from the weighted average of embeddings from multiple prompts.

    Args:
        prompts: List of input prompts
        weights: List of weights for each prompt (will be normalized). If None, equal weights are used.
        max_length: Maximum length of the generated text

    Returns:
        Generated text or None if there's an error
    """
    try:
        if not prompts:
            raise ValueError("No prompts provided")
            
        # Normalize weights if provided, or create equal weights if not
        if weights is None:
            weights = [1.0] * len(prompts)
        else:
            if len(weights) != len(prompts):
                raise ValueError("Number of weights must match number of prompts")
            # Normalize weights to sum to 1
            total = sum(weights)
            weights = [w/total for w in weights]

        # 1. Tokenize all prompts
        all_inputs = [tokenizer(p, return_tensors="pt") for p in prompts]
        
        # Show tokenization
        print("\nTokenization:")
        for i, (prompt, inputs) in enumerate(zip(prompts, all_inputs)):
            print(f"\nPrompt {i+1} '{prompt}' (weight: {weights[i]:.2f}):")
            for id in inputs.input_ids[0]:
                print(f"  {id}: '{tokenizer.decode([id])}'")

        # 2. Find maximum length and pad all inputs
        max_len = max(inputs.input_ids.size(1) for inputs in all_inputs)
        
        # Pad and move to device
        for i in range(len(all_inputs)):
            len_i = all_inputs[i].input_ids.size(1)
            if len_i < max_len:
                pad_length = max_len - len_i
                # Create padding tensor directly on the correct device
                padding = torch.full(
                    (1, pad_length), 
                    tokenizer.pad_token_id, 
                    dtype=torch.long,
                    device=device  # Create tensor directly on the target device
                )
                # Move input to device before concatenating
                all_inputs[i].input_ids = all_inputs[i].input_ids.to(device)
                all_inputs[i].input_ids = torch.cat([all_inputs[i].input_ids, padding], dim=1)
            else:
                # If no padding needed, just move to device
                all_inputs[i].input_ids = all_inputs[i].input_ids.to(device)

        print(f"\nAfter padding, all inputs padded to length: {max_len}")

        # 3. Get embeddings for all prompts
        with torch.no_grad():
            all_embeddings = [
                model.model.embed_tokens(inputs.input_ids) * weight
                for inputs, weight in zip(all_inputs, weights)
            ]

        # 4. Average the embeddings (weighted sum since weights sum to 1)
        averaged_embeddings = sum(all_embeddings)

        # 5. Generate text from averaged embeddings
        with torch.no_grad():
            attention_mask = torch.ones(
                (averaged_embeddings.shape[0], averaged_embeddings.shape[1]), 
                dtype=torch.long,
                device=device
            )
            
            outputs = model.generate(
                inputs_embeds=averaged_embeddings,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )

        # 6. Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error during text generation: {e}")
        return None

if __name__ == "__main__":
    print("Model and tokenizer loaded. You can now interactively generate text by calling:")
    print("generate_text_from_weighted_embeddings(prompts=[...], weights=[...], max_length=...)")
    print("\nRunning example:")
    example_result = generate_text_from_weighted_embeddings(
        prompts=['The cat sat on the', 'The dog ran in the', 'The bird flew over the'],
        weights=[0.4, 0.4, 0.2],
        max_length=60
    )
    print("\nExample result:")
    print(f"'{example_result}'")
    print("\nYou can now try your own prompts!")
    print("Type 'exit()' or press Ctrl+D to quit.")

    # Start an interactive Python REPL
    code.interact(local=locals())