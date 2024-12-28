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

def generate_text(model, tokenizer, input_text, seed, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """Generates text with a given seed and parameters."""
    set_seed(seed)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Avoids warning for sequences without an explicit pad token
    )
    end_time = time.time()
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, end_time - start_time

def test_seed_reproducibility(model, tokenizer, input_text, seed, num_runs=3):
    """Tests if the seed produces reproducible results."""
    logger.info(f"Testing reproducibility with seed: {seed}")
    generations = []
    times = []
    for _ in range(num_runs):
        generated_text, gen_time = generate_text(model, tokenizer, input_text, seed)
        generations.append(generated_text)
        times.append(gen_time)
        logger.info(f"  Run {_ + 1}: {gen_time:.4f}s - {generated_text}")

    # Check if all generations are identical
    all_same = all(gen == generations[0] for gen in generations)
    if all_same:
        logger.info("  Reproducibility test: PASSED")
    else:
        logger.error("  Reproducibility test: FAILED")
        for i, gen in enumerate(generations):
            logger.error(f"    Run {i + 1}: {gen}")

    return generations, times

def test_different_seeds(model, tokenizer, input_text, seeds=[42, 43, 44]):
    """Tests generation with different seeds."""
    logger.info("Testing with different seeds:")
    generations = {}
    times = {}
    for seed in seeds:
        generations[seed], times[seed] = test_seed_reproducibility(model, tokenizer, input_text, seed)

    # Check if generations with different seeds are different
    unique_generations = set(generations[seeds[0]])
    for seed in seeds[1:]:
        unique_generations.update(generations[seed])

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
        generated_text, gen_time = generate_text(model, tokenizer, input_text, seed, **params)
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def test_random_inputs(model, tokenizer, num_tests=3, seed=42):
    """Tests generation with random inputs."""
    logger.info("Testing with random inputs:")
    random.seed(seed)
    for i in range(num_tests):
        random_length = random.randint(10, 50)
        random_input = " ".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random_length))
        logger.info(f"  Test {i + 1}: Input: '{random_input}'")
        generated_text, gen_time = generate_text(model, tokenizer, random_input, seed)
        logger.info(f"    Output: {gen_time:.4f}s - {generated_text}")

def main():
    """Main function to run the demo suite."""
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"  # You can change this to a larger model if desired
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    input_text = "The quick brown fox"

    # Run tests
    test_seed_reproducibility(model, tokenizer, input_text, seed=42)
    test_different_seeds(model, tokenizer, input_text)
    test_parameter_influence(model, tokenizer, input_text)
    test_random_inputs(model, tokenizer)

if __name__ == "__main__":
    main()