import logging
import asyncio
import httpx
import json
from typing import Dict, Any, List, Optional, Tuple
from rich import print
from rich.table import Table
from rich.console import Console
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Rich Console
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

class CacherClientV2:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(follow_redirects=True)
        logger.info(f"Initialized CacherClientV2 with base URL: {base_url}")

    async def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and logging."""
        endpoint = f"/{endpoint.strip('/')}/"
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.request(method, url, json=json_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API Error: {str(e)}")
            logger.error(f"Error Details: {e.response.json() if e.response else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    async def create_state(self, model_id: str, text: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new state with visualization of token probabilities."""
        logger.info(f"Creating state for model {model_id} with text: {text[:50]}...")
        
        data = {
            "model_id": model_id,
            "text": text,
            "config": config or {}
        }
        
        response = await self._make_request("POST", "states", data)
        state_id = response["state_id"]
        
        # Visualize token probabilities
        table = Table(title="Token Analysis for Initial State")
        table.add_column("Position", justify="right", style="cyan")
        table.add_column("Token", style="magenta")
        table.add_column("Probability", justify="right", style="green")
        
        if "token_analysis" in response:
            for pos, token_info in enumerate(response["token_analysis"]):
                table.add_row(
                    str(pos),
                    token_info["token"],
                    f"{token_info['probability']:.4f}"
                )
            console.print(table)
        
        return response

    async def generate_tokens(
        self,
        state_id: str,
        max_tokens: int = 1,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate tokens with visualization of generation process."""
        logger.info(f"Generating {max_tokens} tokens for state {state_id}")
        
        data = {
            "state_id": state_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        if stream:
            async with self.client.stream("POST", f"{self.base_url}/generate/token/", json=data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        token_info = json.loads(line)
                        
                        # Create token probability table
                        table = Table(title=f"Token Generation Step {token_info['step']}")
                        table.add_column("Token", style="magenta")
                        table.add_column("Probability", justify="right", style="green")
                        table.add_column("Entropy", justify="right", style="red")
                        
                        for token, prob in token_info["top_tokens"]:
                            entropy = -prob * np.log2(prob) if prob > 0 else 0
                            table.add_row(
                                token,
                                f"{prob:.4f}",
                                f"{entropy:.4f}"
                            )
                        
                        console.print(table)
                        print(token_info["token"], end="", flush=True)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {str(e)}, line: {line}")
                        continue
            
            return {"status": "stream complete"}
        else:
            return await self._make_request("POST", "generate/token", data)

    async def generate_beam(
        self,
        state_id: str,
        num_beams: int = 5,
        max_length: int = 50,
        diversity_penalty: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate text using beam search with visualization."""
        logger.info(f"Starting beam search for state {state_id}")
        
        data = {
            "state_id": state_id,
            "num_beams": num_beams,
            "max_length": max_length,
            "diversity_penalty": diversity_penalty
        }
        
        response = await self._make_request("POST", "generate/beam", data)
        
        # Visualize beam search results
        table = Table(title="Beam Search Results")
        table.add_column("Beam", justify="right", style="cyan")
        table.add_column("Text", style="magenta")
        table.add_column("Score", justify="right", style="green")
        
        for i, beam in enumerate(response["beams"], 1):
            table.add_row(
                f"#{i}",
                beam["text"],
                f"{beam['score']:.4f}"
            )
        
        console.print(table)
        return response

    async def analyze_attention(self, state_id: str):
        """Analyze attention patterns with visualization."""
        logger.info(f"Analyzing attention patterns for state {state_id}")
        
        data = {
            "layer": 0,
            "threshold": 0.1,
            "head": 0
        }
        
        response = await self._make_request("POST", f"analysis/attention/{state_id}", json_data=data)
        
        # Create attention heatmap
        attention_matrix = np.array(response["attention_matrix"])
        tokens = response["tokens"]
        metadata = response["metadata"]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_matrix, annot=True, fmt=".2f", cmap='viridis',
                   xticklabels=tokens, yticklabels=tokens)
        plt.title(f"Attention Patterns (Layer {response['layer']}, Head {response['head']})\n" +
                 f"Model: {metadata['model']}, Threshold: {metadata['threshold']}")
        plt.xlabel("Target Tokens")
        plt.ylabel("Source Tokens")
        plt.tight_layout()
        plt.savefig("plots/attention_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention analysis complete - saved plot to plots/attention_heatmap.png\n" +
                   f"Max attention: {metadata['max_attention']:.3f}, " +
                   f"Min attention: {metadata['min_attention']:.3f}")

    async def inspect_cache(self, state_id: str):
        """Inspect key/value cache with visualizations."""
        logger.info(f"Inspecting cache for state {state_id}")
        
        response = await self._make_request("POST", f"analysis/cache/{state_id}")
        
        # Simulate cache data
        cache_data = response.get("cache_data", {
            "keys": np.random.rand(5, 10),
            "values": np.random.rand(5, 10),
            "layers": ["Layer " + str(i) for i in range(5)]
        })
        
        # Plot key similarities
        plt.figure(figsize=(12, 4))
        key_similarities = cosine_similarity(cache_data["keys"])
        sns.heatmap(key_similarities, annot=True, cmap='coolwarm')
        plt.title("Key Similarities Across Layers")
        plt.savefig("plots/key_similarities.png")
        plt.close()
        
        # Plot value distributions
        plt.figure(figsize=(12, 4))
        plt.boxplot([layer_values for layer_values in cache_data["values"]], 
                   labels=cache_data["layers"])
        plt.title("Value Distributions by Layer")
        plt.ylabel("Value Magnitude")
        plt.savefig("plots/value_distributions.png")
        plt.close()
        
        logger.info("Cache inspection complete - saved plots to plots/")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def run_demo():
    """Run a comprehensive demo of the Cacher API v3 capabilities."""
    client = CacherClientV2()
    logger.info("Starting Enhanced Cacher API Demo")

    try:
        # 1. Create a state with initial text
        print("\n[bold blue]1. Creating Initial State[/bold blue]")
        state = await client.create_state(
            "gpt2",
            "Once upon a time in a magical forest,",
            config={"temperature": 0.7, "top_p": 0.9}
        )
        state_id = state["state_id"]
        logger.info(f"Created state with ID: {state_id}")

        # 2. Generate tokens with streaming
        print("\n[bold blue]2. Generating Tokens with Streaming[/bold blue]")
        await client.generate_tokens(
            state_id,
            max_tokens=10,
            temperature=0.8,
            stream=True
        )

        # 3. Generate text using beam search
        print("\n[bold blue]3. Generating Text with Beam Search[/bold blue]")
        beam_results = await client.generate_beam(
            state_id,
            num_beams=5,
            max_length=30,
            diversity_penalty=0.5
        )
        logger.info(f"Generated {len(beam_results['beams'])} diverse continuations")

        # 4. Analyze attention patterns
        print("\n[bold blue]4. Analyzing Attention Patterns[/bold blue]")
        await client.analyze_attention(state_id)
        logger.info("Attention analysis complete")

        # 5. Inspect the cache
        print("\n[bold blue]5. Inspecting Key/Value Cache[/bold blue]")
        await client.inspect_cache(state_id)
        logger.info("Cache inspection complete")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
    finally:
        await client.close()


def main():
    """Entry point for the demo."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error running demo: {str(e)}")


if __name__ == "__main__":
    main() 