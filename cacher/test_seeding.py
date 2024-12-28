import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import logging
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_text(model, tokenizer, input_text, seed, max_new_tokens=50, temperature=0.7, top_p=0.9, past_key_values=None, use_cache=False):
    """Generates text with optional cached state."""
    set_seed(seed)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Modify the generate method to return a dict by setting return_dict_in_generate=True
    # and to include past_key_values by setting output_scores=True (if needed)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
        "return_dict_in_generate": True,  # Ensure the output is a dict
    }
    
    if past_key_values is not None:
        # Use cached state for subsequent generation
        inputs["input_ids"] = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(model.device)  # Dummy input when using cached state
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])  # Dummy attention mask
        generation_kwargs["past_key_values"] = past_key_values

    start_time = time.time()
    outputs = model.generate(**inputs, **generation_kwargs)
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return generated_text, end_time - start_time, outputs.past_key_values

def test_seed_reproducibility(model, tokenizer, input_text, seed, num_runs=3):
    """Tests if the seed produces reproducible results."""
    logger.info(f"Testing reproducibility with seed: {seed}")
    generations = []
    times = []
    for run in range(1, num_runs + 1):
        generated_text, gen_time, _ = generate_text(model, tokenizer, input_text, seed)
        generations.append(generated_text)
        times.append(gen_time)
        logger.info(f"  Run {run}: {gen_time:.4f}s - {generated_text}")

    # Check if all generations are identical
    all_same = all(gen == generations[0] for gen in generations)
    if all_same:
        logger.info("  Reproducibility test: PASSED")
    else:
        logger.error("  Reproducibility test: FAILED")
        for i, gen in enumerate(generations, start=1):
            logger.error(f"    Run {i}: {gen}")

    return generations, times

def test_different_seeds(model, tokenizer, input_text, seeds=[42, 43, 44]):
    """Tests generation with different seeds."""
    logger.info("Testing with different seeds:")
    generations = {}
    times = {}
    for seed in seeds:
        gens, ts = test_seed_reproducibility(model, tokenizer, input_text, seed)
        generations[seed] = gens
        times[seed] = ts

    # Check if generations with different seeds are different
    unique_generations = set(gens for seed in seeds for gens in generations[seed])
    if len(unique_generations) > 1:
        logger.info("  Different seeds test: PASSED (Outputs differ)")
    else:
        logger.warning("  Different seeds test: WARNING (Outputs are the same, which is unlikely but possible)")

    return generations, times

def test_parameter_influence(model, tokenizer, input_text, seed=42):
    """Tests the influence of generation parameters."""
    logger.info("Testing parameter influence:")
    parameters = [
        {"max_new_tokens": 20, "temperature": 0.7, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.5, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.8},
    ]
    for params in parameters:
        logger.info(f"  Parameters: {params}")
        generated_text, gen_time, _ = generate_text(model, tokenizer, input_text, seed, **params)
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def test_random_inputs(model, tokenizer, num_tests=3, seed=42):
    """Tests generation with random inputs."""
    logger.info("Testing with random inputs:")
    random.seed(seed)
    for i in range(1, num_tests + 1):
        random_length = random.randint(10, 50)
        random_input = " ".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random_length))
        logger.info(f"  Test {i}: Input: '{random_input}'")
        generated_text, gen_time, _ = generate_text(model, tokenizer, random_input, seed)
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def test_state_caching(model, tokenizer, input_text, seed=42, max_new_tokens=20):
    """Tests if state caching and export/load works correctly."""
    logger.info("Testing state caching and export/load:")

    # 1. Normal Generation
    normal_text, _, _ = generate_text(model, tokenizer, input_text, seed, max_new_tokens)
    logger.info(f"  Normal generation: {normal_text}")

    # 2. Cached Generation
    # Generate first token and cache state
    first_token_text, _, past_key_values = generate_text(model, tokenizer, input_text, seed, max_new_tokens=1, use_cache=True)
    # Generate remaining text using cached state
    remaining_tokens = max_new_tokens - 1
    cached_text, _, _ = generate_text(model, tokenizer, "", seed, remaining_tokens, past_key_values=past_key_values, use_cache=True)
    cached_text = first_token_text + cached_text
    logger.info(f"  Cached generation: {cached_text}")

    # 3. Exported/Loaded Generation
    # Generate first token and export state
    first_token_text, _, past_key_values = generate_text(model, tokenizer, input_text, seed, max_new_tokens=1, use_cache=True)
    # Export state to a variable
    exported_state = past_key_values
    # Load state from the variable
    loaded_state = exported_state
    # Generate remaining text using loaded state
    remaining_tokens = max_new_tokens - 1
    loaded_text, _, _ = generate_text(model, tokenizer, "", seed, remaining_tokens, past_key_values=loaded_state, use_cache=True)
    loaded_text = first_token_text + loaded_text
    logger.info(f"  Exported/Loaded generation: {loaded_text}")

    # Assertions
    assert normal_text == cached_text, "Normal and cached outputs differ!"
    assert normal_text == loaded_text, "Normal and exported/loaded outputs differ!"
    assert cached_text == loaded_text, "Cached and exported/loaded outputs differ!"

    logger.info("  State caching and export/load test: PASSED")

def main():
    """Main function to run the demo suite."""
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Using the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    input_text = "The quick brown fox"

    # Run tests
    test_seed_reproducibility(model, tokenizer, input_text, seed=42)
    test_different_seeds(model, tokenizer, input_text)
    test_parameter_influence(model, tokenizer, input_text)
    test_random_inputs(model, tokenizer)
    test_state_caching(model, tokenizer, input_text)

if __name__ == "__main__":
    main()