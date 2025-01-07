import logging
import asyncio
import json
import httpx
from rich import print
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time

# Configure matplotlib to use a specific font and reduce logging
plt.rcParams['font.family'] = 'DejaVu Sans'

# Silence matplotlib font debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Initialize Rich console and configure logging with custom format
console = Console()
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            console=console,
            show_time=False
        )
    ]
)
logger = logging.getLogger(__name__)

class CacherClientV3:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(follow_redirects=True)
        self.first_run_time = 0
        self.second_run_time = 0
        logger.info(f"Initialized CacherClientV3 with base URL: {base_url}")

    async def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and logging."""
        endpoint = f"/{endpoint.strip('/')}"
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Making {method} request to {url} with data: {json_data}")
        try:
            response = await self.client.request(method, url, json=json_data)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            else:
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

    async def create_state(self, model_id: str, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new state."""
        data = {"model_id": model_id, "text": text, "config": config or {}}
        return await self._make_request("POST", "/states", json=data)

    async def get_state(self, state_id: str) -> Dict[str, Any]:
        """Get state by ID."""
        response = await self._make_request("GET", f"/states/{state_id}")
        return response.json()

    async def delete_state(self, state_id: str) -> Dict[str, Any]:
        """Delete state by ID."""
        response = await self._make_request("DELETE", f"/states/{state_id}")
        return response.json()

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
        return response.json()

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

    async def analyze_tokens(self, state_id: str) -> Dict[str, Any]:
        """Analyze token-level information for a state."""
        response = await self._make_request("GET", f"/analysis/tokens/{state_id}")
        return response.json()

    async def analyze_attention(self, state_id: str, layer: Optional[int] = None, threshold: Optional[float] = 0.1, head: Optional[int] = None) -> Dict[str, Any]:
        """Analyze attention patterns for a specific state and layer."""
        data = {
            "state_id": state_id,
            "layer": layer,
            "threshold": threshold,
            "head": head,
        }
        response = await self._make_request("POST", "/analysis/attention", json=data)
        return response.json()

    async def inspect_cache(self, state_id: str, key_name: Optional[str] = None, layer_range: Optional[List[int]] = None) -> Dict[str, Any]:
        """Inspect the attention cache for a given state and layer."""
        data = {
            "state_id": state_id,
            "key_name": key_name,
            "layer_range": layer_range,
        }
        response = await self._make_request("POST", "/analysis/cache/inspect", json=data)
        return response.json()

    async def _process_message(self, state_id: str, message: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Processes a single message in the conversation."""
        start_time = asyncio.get_event_loop().time()
        try:
            if message["role"] == "user":
                # Generate response using the current state
                response = await self.generate_token(
                    state_id=state_id,
                    max_tokens=50,
                    temperature=0.7
                )
                
                # Create new state for next turn with the user's message
                new_state = await self.create_state(
                    "gpt2", 
                    message["content"],
                    config={"temperature": 0.7}
                )
                
                return {
                    "content": response.get("text", ""),
                    "role": "assistant",
                    "state_id": new_state["state_id"]
                }
            elif message["role"] == "assistant":
                return None
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            if hasattr(e, 'response'):
                try:
                    error_details = e.response.json()
                except json.JSONDecodeError:
                    error_details = e.response.text if e.response else 'No response'
                logger.error(f"Error Details: {error_details}")
            return None
        finally:
            end_time = asyncio.get_event_loop().time()
            elapsed_time = end_time - start_time
            logger.info(f"Time to process message: {elapsed_time:.4f} seconds")

    async def demonstrate_conversation_caching(self):
        """Demonstrates caching in a conversational context."""
        logger.info("Starting conversation caching demonstration")
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well, thank you for asking."},
            {"role": "user", "content": "What can you do?"},
        ]

        logger.info("First Run (No Caching):")
        first_run_start_time = asyncio.get_event_loop().time() # Start timer for first run
        start_state = await self.create_state("gpt2", messages[0]["content"])
        current_state_id = start_state["state_id"]
        logger.info(f"Created state: {current_state_id} with message: {messages[0]['content']}")
        for message in messages[1:]:
            logger.info(f"Processing message: {message['content']}")
            response = await self._process_message(current_state_id, message)
            if response and "state_id" in response:
                current_state_id = response["state_id"]
                logger.info(f"New state id: {current_state_id}")
            elif message["role"] == "assistant":
                logger.info("Assistant responded, no new state created yet.")
        first_run_end_time = asyncio.get_event_loop().time() # End timer for first run
        self.first_run_time = first_run_end_time - first_run_start_time # Calculate first run time

        logger.info("Second Run (Potential Caching):")
        second_run_start_time = asyncio.get_event_loop().time() # Start timer for second run
        start_state_cached = await self.create_state("gpt2", messages[0]["content"])
        current_state_id_cached = start_state_cached["state_id"]
        logger.info(f"Created state: {current_state_id_cached} with message: {messages[0]['content']}")
        for message in messages[1:]:
            logger.info(f"Processing message: {message['content']}")
            response_cached = await self._process_message(current_state_id_cached, message)
            if response_cached and "state_id" in response_cached:
                current_state_id_cached = response_cached["state_id"]
                logger.info(f"New state id: {current_state_id_cached}")
            elif message["role"] == "assistant":
                logger.info("Assistant responded, no new state created yet.")
        second_run_end_time = asyncio.get_event_loop().time() # End timer for second run
        self.second_run_time = second_run_end_time - second_run_start_time # Calculate second run time

        logger.info(f"Total time for first run (uncached): {self.first_run_time:.4f} seconds")
        logger.info(f"Total time for second run (cached): {self.second_run_time:.4f} seconds")

        # Clean up
        await self.delete_state(start_state["state_id"])
        await self.delete_state(start_state_cached["state_id"])

    async def run_demo(self):
        """Run a comprehensive demo of the Cacher API capabilities."""
        logger.info("Starting Enhanced Caching Demo")
        start_state = await self.create_state("gpt2", "The quick brown fox")
        logger.info(f"Created initial state: {start_state['state_id']}")

        # Demonstrate conversation caching
        await self.demonstrate_conversation_caching()

        # Clean up
        await self.delete_state(start_state["state_id"])
        logger.info(f"Deleted initial state: {start_state['state_id']}")

    async def close(self):
        await self.client.aclose()

    async def get_top_n_tokens(self, state_id: str, n: int = 5) -> Dict[str, Any]:
        """Retrieves the top N tokens and their probabilities for the next token."""
        return await self._make_request("GET", f"/states/{state_id}/top_tokens?n={n}")

async def main():
    client = CacherClientV3()
    try:
        await client.run_demo()
    finally:
        await client.close()

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())