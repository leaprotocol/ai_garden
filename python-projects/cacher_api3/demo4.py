import asyncio
import time
import logging
from typing import List, Dict, Optional, Any
import numpy as np
import random
import torch
from transformers import set_seed
from rich.console import Console
from rich.table import Table
import httpx
import json

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

class CacherClientV4:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0  # Add reasonable timeout
        )
        self.first_run_time = 0
        self.second_run_time = 0
        logger.debug(f"Initializing client with base URL: {base_url}")
        logger.info(f"Initialized CacherClientV4 with base URL: {base_url}")

    async def _make_request(self, method: str, endpoint: str, *, json: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and logging."""
        endpoint = f"/{endpoint.strip('/')}"
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making {method} request to {url} with data: {json}")
        try:
            # Use regular request instead of stream for non-streaming responses
            response = await self.client.request(method, url, json=json)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            return {"status": "success", "text": response.text}
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            if e.response is not None:
                try:
                    error_json = e.response.json()
                    logger.error(f"Error details: {error_json}")
                except json.JSONDecodeError:
                    logger.error(f"Response content: {e.response.text}")
            raise

    async def delete_state(self, state_id: str) -> Dict[str, Any]:
        """Delete state by ID."""
        logger.debug(f"Deleting state with ID: {state_id}")
        return await self._make_request("DELETE", f"/states/{state_id}")

    async def generate_token(
        self, state_id: str, max_tokens: int = 1, temperature: float = 1.0, top_p: float = 1.0, stop_sequences: Optional[List[str]] = None, stream: bool = False, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate tokens for a state."""
        data = {
            "state_id": state_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "stream": stream,
            "seed": seed,
        }
        response = await self._make_request("POST", "/generate/token", json=data)
        return response

    async def generate_beam(self, state_id: str, num_beams: int = 5, max_length: int = 50, diversity_penalty: Optional[float] = None, early_stopping: Optional[bool] = True) -> Dict[str, Any]:
        """Generate text using beam search."""
        data = {
            "state_id": state_id,
            "num_beams": num_beams,
            "max_length": max_length,
            "diversity_penalty": diversity_penalty,
            "early_stopping": early_stopping,
        }
        response = await self._make_request("POST", "/generate/beam", json=data)
        return response.json()

    async def create_state(self, model_id: str, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new state."""
        data = {"model_id": model_id, "text": text, "config": config or {}}
        return await self._make_request("POST", "/states", json=data)

    async def get_top_n_tokens(self, state_id: str, n: int = 5) -> Dict[str, Any]:
        """Retrieves the top N tokens and their probabilities for the next token."""
        return await self._make_request("GET", f"/states/{state_id}/top_tokens?n={n}")

    async def close(self):
        await self.client.aclose()

async def main():
    client = CacherClientV4()
    logger.info("Starting Comprehensive Cacher API Demo (demo4.py)")

    try:
        # Configure logging for detailed debugging
        logging.getLogger('httpx').setLevel(logging.DEBUG)
        logger.debug("HTTP client debugging enabled")

        # 1. Basic Text Generation with Configurable Parameters
        print("\n[bold blue]1. Basic Text Generation with Parameters[/bold blue]")
        initial_text = "The quick brown fox jumps over the lazy"
        temperature = 0.7
        top_p = 0.9
        logger.info(f"Creating state with initial text: '{initial_text}', temperature: {temperature}, top_p: {top_p}")
        state = await client.create_state("gpt2", initial_text, config={"temperature": temperature, "top_p": top_p})
        state_id = state["state_id"]
        logger.info(f"Created state with ID: {state_id}")

        logger.info("Generating the next 10 tokens...")
        generation_result = await client.generate_token(state_id, max_tokens=10)
        logger.info(f"Generated text: '{generation_result['text']}'")

        # 2. Branching and Caching
        print("\n[bold blue]2. Branching and Caching[/bold blue]")
        logger.info(f"Generating a second continuation from the same state (cached)...")
        start_time = time.time()
        generation_result_cached = await client.generate_token(state_id, max_tokens=10)
        end_time = time.time()
        logger.info(f"Generated text (cached): '{generation_result_cached['text']}' in {end_time - start_time:.4f} seconds")

        # Generate a third continuation to compare uncached time
        print("\n[bold blue]3. Uncached Generation for Comparison[/bold blue]")
        logger.info("Generating from a new state to compare uncached time...")
        start_time_uncached = time.time()
        new_state = await client.create_state("gpt2", initial_text, config={"temperature": temperature, "top_p": top_p})
        new_state_id = new_state["state_id"]
        generation_result_uncached = await client.generate_token(new_state_id, max_tokens=10)
        end_time_uncached = time.time()
        logger.info(f"Generated text (uncached): '{generation_result_uncached['text']}' in {end_time_uncached - start_time_uncached:.4f} seconds")

        # 4. State Management: Deleting a State
        print("\n[bold blue]4. State Management: Deleting a State[/bold blue]")
        logger.info(f"Deleting state with ID: {state_id}")
        delete_result = await client.delete_state(state_id)
        logger.info(f"Deletion successful: {delete_result}")

        # 5. Reproducibility with Seeding
        print("\n[bold blue]5. Reproducibility with Seeding[/bold blue]")
        seed = 42
        set_seed(seed)
        logger.info(f"Testing reproducibility with seed: {seed}")
        
        # First generation
        logger.info("First generation...")
        seeded_state_1 = await client.create_state("gpt2", initial_text, config={"seed": seed})
        generation_seeded_1 = await client.generate_token(
            seeded_state_1["state_id"],
            max_tokens=5,
            seed=seed,
            temperature=0.7,
            top_p=0.9
        )
        logger.info(f"First generation with seed: '{generation_seeded_1['text']}'")

        # Second generation with same parameters
        logger.info("Second generation with same seed...")
        seeded_state_2 = await client.create_state("gpt2", initial_text, config={"seed": seed})
        generation_seeded_2 = await client.generate_token(
            seeded_state_2["state_id"],
            max_tokens=5,
            seed=seed,
            temperature=0.7,
            top_p=0.9
        )
        logger.info(f"Second generation with seed: '{generation_seeded_2['text']}'")

        # Verify reproducibility
        if generation_seeded_1['text'] == generation_seeded_2['text']:
            logger.info("[bold green]Reproducibility test passed![/bold green]")
        else:
            logger.warning("[yellow]Reproducibility test failed (output mismatch).[/yellow]")
            logger.warning(f"First output:  {generation_seeded_1['text']}")
            logger.warning(f"Second output: {generation_seeded_2['text']}")

        # 6. Simplified Token Probability Visualization
        print("\n[bold blue]6. Token Probability Visualization[/bold blue]")
        top_tokens_info = await client.get_top_n_tokens(new_state_id, n=5)

        if top_tokens_info:
            table = Table(title="Top 5 Predicted Tokens")
            table.add_column("Token", style="magenta")
            table.add_column("Probability", justify="right", style="green")

            for token_data in top_tokens_info.get('tokens', []):
                table.add_row(token_data['token'], f"{token_data['probability']:.4f}")
            console.print(table)
        else:
            logger.warning("Could not retrieve top tokens.")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources...")
        await client.close()
        logger.info("Demo completed")

if __name__ == "__main__":
    asyncio.run(main()) 