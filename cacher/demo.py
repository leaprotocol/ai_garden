# demo.py
from state_cacher import StateCacher
import torch
import logging
import sys
import os
import time
import psutil
import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Generation seeds configuration
GENERATION_SEEDS = [111, 222, 222]  # First different, last two same

def set_generation_seed(seed: int):
    """Set seed for generation specifically."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class BenchmarkResult:
    """Store benchmark results for a single generation approach."""
    method: str
    total_time: float
    peak_memory: float
    outputs: List[str]
    tokens_per_second: float
    per_generation_times: List[float]

def get_memory_usage() -> float:
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def format_chat_message(message: str) -> str:
    """Format a message in the chat format expected by SmolLM2."""
    if not message.startswith("User: "):
        message = f"User: {message}"
    if not message.endswith("\nAssistant:"):
        message = f"{message}\nAssistant:"
    return message

def generate_with_seed(
    cacher: StateCacher,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    seed: int,
    max_new_tokens: int = 150,
    use_cache: bool = True
) -> torch.Tensor:
    """Generate text with a specific seed."""
    set_generation_seed(seed)
    with torch.no_grad():
        outputs = cacher.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=cacher.tokenizer.pad_token_id,
            use_cache=use_cache
        )
    return outputs

def run_cached_benchmark(
    cacher: StateCacher,
    initial_text: str,
    suffixes: List[str],
    max_new_tokens: int = 150
) -> BenchmarkResult:
    """Run benchmark using the cached approach."""
    logger.info("\nRunning cached benchmark...")
    start_memory = get_memory_usage()
    peak_memory = start_memory
    start_time = time.time()
    per_generation_times = []
    
    # Initial processing and caching
    generated_text, cached_state = cacher.process_and_cache(initial_text)
    
    # Generate continuations using cached state
    continuations = []
    total_tokens = 0
    
    for i, suffix in enumerate(suffixes):
        gen_start_time = time.time()
        
        # Prepare inputs
        suffix_inputs = cacher.tokenizer(
            suffix,
            return_tensors="pt"
        ).to(cacher.device)
        
        # Concatenate with cached state
        new_input_ids = torch.cat([cached_state['input_ids'], suffix_inputs.input_ids], dim=1)
        new_attention_mask = torch.cat([cached_state['attention_mask'], suffix_inputs.attention_mask], dim=1)
        
        # Generate with specific seed
        outputs = generate_with_seed(
            cacher,
            new_input_ids,
            new_attention_mask,
            GENERATION_SEEDS[i],
            max_new_tokens,
            use_cache=True
        )
        
        continuation = cacher.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - gen_start_time
        per_generation_times.append(gen_time)
        
        continuations.append(continuation)
        total_tokens += len(cacher.tokenizer.encode(continuation))
        
        # Track peak memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)
        
        logger.info(f"Generation {len(continuations)} (seed={GENERATION_SEEDS[i]}) took {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return BenchmarkResult(
        method="cached",
        total_time=total_time,
        peak_memory=peak_memory - start_memory,
        outputs=continuations,
        tokens_per_second=tokens_per_second,
        per_generation_times=per_generation_times
    )

def run_non_cached_benchmark(
    cacher: StateCacher,
    initial_text: str,
    suffixes: List[str],
    max_new_tokens: int = 150
) -> BenchmarkResult:
    """Run benchmark using the non-cached approach."""
    logger.info("\nRunning non-cached benchmark...")
    start_memory = get_memory_usage()
    peak_memory = start_memory
    start_time = time.time()
    per_generation_times = []
    
    continuations = []
    total_tokens = 0
    
    for i, suffix in enumerate(suffixes):
        gen_start_time = time.time()
        
        # Combine initial text with suffix
        full_text = initial_text.rstrip("\nAssistant:") + " " + suffix + "\nAssistant:"
        
        # Process without caching
        inputs = cacher.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(cacher.device)
        
        # Generate with same seed as cached version
        outputs = generate_with_seed(
            cacher,
            inputs.input_ids,
            inputs.attention_mask,
            GENERATION_SEEDS[i],
            max_new_tokens,
            use_cache=False
        )
        
        continuation = cacher.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - gen_start_time
        per_generation_times.append(gen_time)
        
        continuations.append(continuation)
        total_tokens += len(cacher.tokenizer.encode(continuation))
        
        # Track peak memory
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)
        
        logger.info(f"Generation {len(continuations)} (seed={GENERATION_SEEDS[i]}) took {gen_time:.2f}s")
    
    total_time = time.time() - start_time
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return BenchmarkResult(
        method="non-cached",
        total_time=total_time,
        peak_memory=peak_memory - start_memory,
        outputs=continuations,
        tokens_per_second=tokens_per_second,
        per_generation_times=per_generation_times
    )

def compare_outputs(cached_outputs: List[str], non_cached_outputs: List[str]) -> Dict[str, float]:
    """Compare the similarity between cached and non-cached outputs."""
    from difflib import SequenceMatcher
    
    similarities = []
    for i, (c_out, nc_out) in enumerate(zip(cached_outputs, non_cached_outputs)):
        similarity = SequenceMatcher(None, c_out, nc_out).ratio()
        similarities.append(similarity)
        if similarity < 1.0:  # If not identical
            logger.warning(f"Generation {i+1} outputs differ (similarity: {similarity:.2%})")
            logger.warning(f"Cached: {c_out[:100]}...")
            logger.warning(f"Non-cached: {nc_out[:100]}...")
    
    return {
        "min_similarity": min(similarities),
        "max_similarity": max(similarities),
        "avg_similarity": sum(similarities) / len(similarities)
    }

def print_benchmark_results(cached: BenchmarkResult, non_cached: BenchmarkResult):
    """Print detailed comparison of benchmark results."""
    logger.info("\n=== Benchmark Results ===")
    
    # Time comparison
    speedup = non_cached.total_time / cached.total_time if cached.total_time > 0 else float('inf')
    logger.info(f"\nTime Comparison:")
    logger.info(f"Cached: {cached.total_time:.2f}s")
    logger.info(f"Non-cached: {non_cached.total_time:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Per-generation time comparison
    logger.info(f"\nPer-Generation Times:")
    for i, (cached_time, non_cached_time) in enumerate(zip(cached.per_generation_times, non_cached.per_generation_times), 1):
        speedup = non_cached_time / cached_time if cached_time > 0 else float('inf')
        logger.info(f"Generation {i} (seed={GENERATION_SEEDS[i-1]}):")
        logger.info(f"  Cached: {cached_time:.2f}s")
        logger.info(f"  Non-cached: {non_cached_time:.2f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    # Memory comparison
    logger.info(f"\nPeak Memory Usage:")
    logger.info(f"Cached: {cached.peak_memory:.2f}MB")
    logger.info(f"Non-cached: {non_cached.peak_memory:.2f}MB")
    
    # Performance metrics
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"Cached tokens/sec: {cached.tokens_per_second:.2f}")
    logger.info(f"Non-cached tokens/sec: {non_cached.tokens_per_second:.2f}")
    
    # Output comparison
    similarities = compare_outputs(cached.outputs, non_cached.outputs)
    logger.info(f"\nOutput Similarity:")
    logger.info(f"Min: {similarities['min_similarity']:.2%}")
    logger.info(f"Max: {similarities['max_similarity']:.2%}")
    logger.info(f"Avg: {similarities['avg_similarity']:.2%}")

def main():
    try:
        # Initialize the cacher with SmolLM2
        logger.info("Initializing StateCacher with SmolLM2...")
        cacher = StateCacher(
            model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Longer common prefix with shorter, more focused suffixes
        initial_text = format_chat_message(
            "You are a helpful AI assistant. I want to understand the three main types of machine learning. "
            "Please provide a clear and concise explanation that covers supervised learning, unsupervised learning, "
            "and reinforcement learning. For each type"
        )
        
        # Shorter, more focused suffixes that complete the question naturally
        suffixes = [
            ", explain its key characteristics:",
            ", describe how it works:",
            ", list its main applications:"
        ]
        
        # Run benchmarks
        cached_results = run_cached_benchmark(cacher, initial_text, suffixes)
        non_cached_results = run_non_cached_benchmark(cacher, initial_text, suffixes)
        
        # Print comparison
        print_benchmark_results(cached_results, non_cached_results)
        
        # Print sample outputs for manual comparison
        logger.info("\n=== Sample Output Comparison ===")
        for i, (cached, non_cached) in enumerate(zip(cached_results.outputs, non_cached_results.outputs)):
            logger.info(f"\nGeneration {i+1} (seed={GENERATION_SEEDS[i]}):")
            logger.info(f"Suffix: {suffixes[i]}")
            logger.info(f"Cached output: {cached[:200]}...")
            logger.info(f"Non-cached output: {non_cached[:200]}...")
        
    except KeyboardInterrupt:
        logger.info("\nGracefully shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
