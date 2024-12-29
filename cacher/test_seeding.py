import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import logging
import time
import random
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Initialize Rich console
console = Console()

# Configure logging with RichHandler for colorful output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[RichHandler(console=console, show_time=False, markup=True)]
)
logger = logging.getLogger(__name__)

console.print("[bold cyan]Starting test_seeding.py with Rich formatting[/bold cyan]\n")

def generate_text(
    model,
    tokenizer,
    input_text=None,
    input_ids=None,
    attention_mask=None,
    seed=None,
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
        seed (int, optional): Seed for reproducibility.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        past_key_values (tuple, optional): Cached past key values.
        use_cache (bool): Whether to use cache.

    Returns:
        generated_text (str): The generated text.
        generation_time (float): Time taken for generation.
        past_key_values (tuple or None): The updated cache.
    """
    if seed is not None:
        set_seed(seed)
    else:
        logger.debug("No seed provided; skipping seed setting.")
    
    if input_text:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # Enable sampling if temperature or top_p is set
        "do_sample": True if temperature > 0 or top_p < 1.0 else False,
        "use_cache": use_cache,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    
    if past_key_values is not None:
        generation_kwargs["past_key_values"] = past_key_values
    
    logger.debug(f"Generation kwargs: {generation_kwargs}")
    
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
    end_time = time.time()
    generation_time = end_time - start_time
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    logger.debug(f"Generated text: {generated_text}")
    
    # Extract past_key_values if available
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        past_key_values_out = outputs.past_key_values
    elif hasattr(outputs, 'cache') and outputs.cache is not None:
        past_key_values_out = outputs.cache
    else:
        past_key_values_out = None
    
    return generated_text, generation_time, past_key_values_out

def test_seed_reproducibility(model, tokenizer, input_text, seed, num_runs=3):
    """Tests if the seed produces reproducible results."""
    logger.info(f"Testing reproducibility with seed: [bold]{seed}[/bold]")
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
    if all(gen == generations[0] for gen in generations):
        logger.info("[bold green]  Reproducibility test: PASSED[/bold green]")
    else:
        logger.error("[bold red]  Reproducibility test: FAILED[/bold red]")
    return generations, times

def test_different_seeds(model, tokenizer, input_text, seeds, num_runs=3):
    """Tests generation with different seeds."""
    logger.info("Testing with different seeds:")
    first_seed_output = None
    all_outputs_different = True
    for seed in seeds:
        gens, ts = test_seed_reproducibility(model, tokenizer, input_text, seed, num_runs=num_runs)
        if first_seed_output is None:
            first_seed_output = gens[0]
        elif any(gen == first_seed_output for gen in gens):
            all_outputs_different = False
            break
    
    if all_outputs_different:
        logger.info("[bold green]  Different seeds test: PASSED (Outputs differ)[/bold green]")
    else:
        logger.error("[bold red]  Different seeds test: FAILED (Outputs are the same)[/bold red]")
        logger.error(f"  First seed output: {first_seed_output}")
        logger.error(f"  All outputs: {gens}")

def test_parameter_influence(model, tokenizer, input_text, seed=42):
    """Tests the influence of different generation parameters."""
    logger.info("Testing parameter influence:")
    params = [
        {"max_new_tokens": 20, "temperature": 0.7, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.5, "top_p": 0.9},
        {"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.8},
    ]
    
    table = Table(title="Parameter Influence Test")
    table.add_column("Parameters", style="cyan")
    table.add_column("Output", style="magenta")
    table.add_column("Time (s)", style="green")
    
    for param_set in params:
        logger.info(f"  Parameters: {param_set}")
        start_time = time.time()
        output_text, _, _ = generate_text(model, tokenizer, input_text=input_text, seed=seed, **param_set)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"    Output: {elapsed_time:.4f}s - {output_text}")
        table.add_row(str(param_set), output_text, f"{elapsed_time:.4f}")

    console.print(table)

def test_random_inputs(model, tokenizer, num_tests=3, seed=42):
    """Tests generation with random inputs."""
    logger.info("Testing with random inputs:")
    random.seed(seed)
    for i in range(1, num_tests + 1):
        random_length = random.randint(10, 50)
        random_input = " ".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random_length))
        logger.info(f"  Test {i}: Input: '[bold yellow]{random_input}[/bold yellow]'")
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
    # Generate first part and capture past_key_values
    first_token_text, _, past_key_values = generate_text(
        model, tokenizer, input_text=input_text, seed=seed, max_new_tokens=1, use_cache=True
    )
    logger.info(f"  First token generation: {first_token_text}")

    if past_key_values is not None:
        logger.debug("  Cached Generation - past_key_values shapes:")
        for layer_idx, layer_pkv in enumerate(past_key_values):
            for tensor_idx, tensor in enumerate(layer_pkv):
                logger.debug(f"    Layer {layer_idx}, Tensor {tensor_idx} shape: {tensor.shape}")
    else:
        logger.debug("  Cached Generation - past_key_values is None")

    # Generate continuation WITH passing cached state
    cached_text, _, _ = generate_text(
        model,
        tokenizer,
        input_ids=torch.tensor([tokenizer.encode(first_token_text)]),
        attention_mask=torch.ones(1, len(tokenizer.encode(first_token_text)), dtype=torch.long),
        seed=seed,
        max_new_tokens=max_new_tokens - 1,
        past_key_values=past_key_values,  # Pass the cached state
        use_cache=True
    )
    logger.info(f"  Cached generation: {cached_text}\n")

    # 3. Exported/Loaded Generation
    # Generate continuation WITH passing the same cached state
    start_time = time.time()

    
    # Modify the specific array at past_key_values[2][0][0][0][0] to be all zeros
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        first_layer_output = past_key_values[0][0]
        if isinstance(first_layer_output, torch.Tensor) and first_layer_output.ndim > 1:
            # logger.info("Blindly modified the first element of the first layer output")
            # first_layer_output[0, 0, 0] = 100.0
            # ok if i change this to 0, it will fail the test as expected 
            # i would like to make a test case out of that - no change, pass; change to 0, fails
            pass

    loaded_text, _, _ = generate_text(
        model,
        tokenizer,
        input_ids=torch.tensor([tokenizer.encode(first_token_text)]),
        attention_mask=torch.ones(1, len(tokenizer.encode(first_token_text)), dtype=torch.long),
        seed=seed,
        max_new_tokens=max_new_tokens - 1,
        past_key_values=past_key_values,
        use_cache=True
    )
    elapsed_time = time.time() - start_time
    
    logger.info(str(past_key_values[1][0][0][0][0]))  
    logger.info(f"  Loaded generation took {elapsed_time:.4f} seconds")
    logger.info(f"  Exported/Loaded generation: {loaded_text}\n")

    # Assertions (expecting cached_text and loaded_text to be the same)
    try:
        assert cached_text is not None, "Cached generation failed."
        assert loaded_text is not None, "Exported/Loaded generation failed."
        assert cached_text.strip() == loaded_text.strip(), "Cached and exported/loaded outputs differ!"
        logger.info("[bold green]  State caching and export/load test: PASSED[/bold green]")
    except AssertionError as e:
        logger.error(f"[bold red]  State caching and export/load test: FAILED[/bold red] - {str(e)}\n")

def main():
    """Main function to run the demo suite."""
    try:
        console.print("[bold cyan]Initializing the model and tokenizer...[/bold cyan]")
        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Using the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        console.print("[bold green]Model and tokenizer loaded successfully.[/bold green]\n")
    
        input_text = "System: You are a helpful AI assistant. User: I want to understand the three main types of machine learning. Assistant:"
    
        # Run tests
        # test_seed_reproducibility(model, tokenizer, input_text, seed=42)
        # test_different_seeds(model, tokenizer, input_text, seeds=[42, 43], num_runs=2)
        # test_parameter_influence(model, tokenizer, input_text, seed=42)
        # test_random_inputs(model, tokenizer, num_tests=3, seed=42)
        test_state_caching(model, tokenizer, input_text, seed=42, max_new_tokens=50)
    
    except KeyboardInterrupt:
        logger.info("\n[bold yellow]Gracefully shutting down...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[bold red]An error occurred: {str(e)}[/bold red]", exc_info=True)
        raise

if __name__ == "__main__":
    main()