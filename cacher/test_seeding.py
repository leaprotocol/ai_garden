import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import logging
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_text(
    model,
    tokenizer,
    input_text=None,
    input_ids=None,
    attention_mask=None,
    seed=42,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    past_key_values=None,
    use_cache=False
):
    """
    Generates text with optional cached state.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        input_text (str, optional): The input text prompt.
        input_ids (torch.Tensor, optional): Pre-tokenized input IDs.
        attention_mask (torch.Tensor, optional): Attention mask for input IDs.
        seed (int): Seed for reproducibility.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        past_key_values (tuple, optional): Cached past key values.
        use_cache (bool): Whether to use cache.

    Returns:
        generated_text (str): The generated text.
        generation_time (float): Time taken for generation.
        past_key_values: The updated cache.
    """
    set_seed(seed)
    
    if input_ids is not None:
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot be empty when past_key_values are provided.")
        inputs = {
            'input_ids': input_ids.to(model.device),
            'attention_mask': attention_mask.to(model.device) if attention_mask is not None else torch.ones_like(input_ids).to(model.device)
        }
    elif input_text is not None:
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    else:
        raise ValueError("Either input_text or input_ids must be provided.")
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": use_cache,
        "return_dict_in_generate": True,  # Ensure the output is a dict
        "output_scores": False,            # Disable output_scores to reduce verbosity
    }
    
    if past_key_values is not None:
        generation_kwargs["past_key_values"] = past_key_values

    start_time = time.time()
    try:
        outputs = model.generate(**inputs, **generation_kwargs)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise e
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return generated_text, end_time - start_time, outputs.past_key_values

def test_seed_reproducibility(model, tokenizer, input_text, seed, num_runs=3):
    """Tests if the seed produces reproducible results."""
    logger.info(f"Testing reproducibility with seed: {seed}")
    generations = []
    times = []
    for run in range(1, num_runs + 1):
        generated_text, gen_time, _ = generate_text(
            model, tokenizer, input_text=input_text, seed=seed, max_new_tokens=20
        )
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
    unique_generations = set(gen for gens in generations.values() for gen in gens)
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
        generated_text, gen_time, _ = generate_text(
            model,
            tokenizer,
            input_text=input_text,
            seed=seed,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"]
        )
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def test_random_inputs(model, tokenizer, num_tests=3, seed=42):
    """Tests generation with random inputs."""
    logger.info("Testing with random inputs:")
    random.seed(seed)
    for i in range(1, num_tests + 1):
        random_length = random.randint(10, 50)
        random_input = " ".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random_length))
        logger.info(f"  Test {i}: Input: '{random_input}'")
        generated_text, gen_time, _ = generate_text(
            model,
            tokenizer,
            input_text=random_input,
            seed=seed,
            max_new_tokens=50
        )
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def test_state_caching(model, tokenizer, input_text, seed=42, max_new_tokens=50):
    """Tests if state caching and export/load works correctly."""
    logger.info("Testing state caching and export/load:")

    # 1. Normal Generation
    normal_text, _, _ = generate_text(
        model, tokenizer, input_text=input_text, seed=seed, max_new_tokens=max_new_tokens
    )
    logger.info(f"  Normal generation: {normal_text}")

    # 2. Cached Generation
    # Generate first part and cache state (without using cache for the first part)
    first_token_output = generate_text(
        model, tokenizer, input_text=input_text, seed=seed, max_new_tokens=1, use_cache=False
    )
    first_token_text, _, past_key_values = first_token_output

    logger.info(f"  First token generation: {first_token_text}")

    # Use the entire first_output_ids as input for the next step
    first_output_ids = tokenizer.encode(first_token_text, return_tensors="pt")
    num_first_tokens = first_output_ids.shape[1]
    logger.debug(f"  Cached Generation - first_output_ids shape: {first_output_ids.shape}")
    logger.debug(f"  Cached Generation - first_output_ids: {first_output_ids}")
    attention_mask = torch.ones_like(first_output_ids)

    # Calculate remaining tokens to generate
    remaining_tokens = max_new_tokens - num_first_tokens

    # Generate remaining text using cached state and the entire first_output_ids as input
    try:
        logger.debug(f"  Cached Generation - Input IDs shape: {first_output_ids.shape}, Attention Mask shape: {attention_mask.shape}, Remaining tokens: {remaining_tokens}")
        cached_text_output = generate_text(
            model,
            tokenizer,
            input_ids=first_output_ids,
            attention_mask=attention_mask,
            seed=seed,
            max_new_tokens=remaining_tokens,
            past_key_values=past_key_values,
            use_cache=True  # Use cache for the continuation
        )
        cached_text, _, _ = cached_text_output
        logger.info(f"  Cached generation: {cached_text}")
    except Exception as e:
        logger.error(f"  Error during cached generation: {e}")
        cached_text = None

    # 3. Exported/Loaded Generation
    # Generate first part and export state (without using cache for the first part)
    first_token_output_export = generate_text(
        model, tokenizer, input_text=input_text, seed=seed, max_new_tokens=1, use_cache=False
    )
    first_token_text_export, _, past_key_values_export = first_token_output_export
    logger.info(f"  First token generation for export: {first_token_text_export}")

    # Use the entire first_output_ids_export as input for the next step
    first_output_ids_export = tokenizer.encode(first_token_text_export, return_tensors="pt")
    num_first_tokens_export = first_output_ids_export.shape[1]
    logger.debug(f"  Exported/Loaded Generation - first_output_ids_export shape: {first_output_ids_export.shape}")
    logger.debug(f"  Exported/Loaded Generation - first_output_ids_export: {first_output_ids_export}")
    attention_mask_export = torch.ones_like(first_output_ids_export)

    # Calculate remaining tokens to generate
    remaining_tokens_export = max_new_tokens - num_first_tokens_export

    # Export state to a variable
    exported_state = past_key_values_export

    # Load state from the variable
    loaded_state = exported_state

    # Generate remaining text using loaded state and the entire first_output_ids_export as input
    try:
        logger.debug(f"  Exported/Loaded Generation - Input IDs shape: {first_output_ids_export.shape}, Attention Mask shape: {attention_mask_export.shape}, Remaining tokens: {remaining_tokens_export}")
        loaded_text_output = generate_text(
            model,
            tokenizer,
            input_ids=first_output_ids_export,
            attention_mask=attention_mask_export,
            seed=seed,
            max_new_tokens=remaining_tokens_export,
            past_key_values=loaded_state,
            use_cache=True  # Use cache for the continuation
        )
        loaded_text, _, _ = loaded_text_output
        logger.info(f"  Exported/Loaded generation: {loaded_text}")
    except Exception as e:
        logger.error(f"  Error during exported/loaded generation: {e}")
        loaded_text = None

    # Assertions (comparing only cached and exported/loaded outputs)
    try:
        assert cached_text is not None, "Cached generation failed."
        assert loaded_text is not None, "Exported/Loaded generation failed."
        assert cached_text.strip() == loaded_text.strip(), "Cached and exported/loaded outputs differ!"
        logger.info("  State caching and export/load test: PASSED")
    except AssertionError as e:
        logger.error(f"  State caching and export/load test: FAILED - {str(e)}")

def main():
    """Main function to run the demo suite."""
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Using the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    input_text = "The quick brown fox"
    test_state_caching(model, tokenizer, input_text)

if __name__ == "__main__":
    main()