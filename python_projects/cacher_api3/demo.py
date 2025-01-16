import logging
import asyncio
import httpx
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacherAPIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: int = None, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

class CacherClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")  # Remove trailing slash
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True  # Enable following redirects
        )
        logger.info(f"Initialized CacherClient with base URL: {base_url}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the client session."""
        await self.client.aclose()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            error_details = e.response.json() if e.response else "No response"
            logger.error(f"API Error: {str(e)}")
            logger.error(f"Error Details: {error_details}")
            raise CacherAPIError(
                message=str(e),
                status_code=e.response.status_code if e.response else None,
                details=error_details
            ) from e

    async def create_state(
        self, 
        model_id: str, 
        text: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new state."""
        data = {
            "model_id": model_id,
            "text": text,
            "config": config or {}
        }
        response = await self._make_request("POST", "/states", json=data)
        return response.json()

    async def get_state(self, state_id: str) -> Dict[str, Any]:
        """Get state by ID."""
        response = await self._make_request("GET", f"/states/{state_id}")
        return response.json()

    async def delete_state(self, state_id: str) -> Dict[str, Any]:
        """Delete state by ID."""
        response = await self._make_request("DELETE", f"/states/{state_id}")
        return response.json()

    async def generate_tokens(
        self,
        state_id: str,
        max_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate tokens for a state."""
        data = {
            "state_id": state_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop_sequences,
            "stream": stream
        }
        
        response = await self._make_request("POST", "/generate/token", json=data)
        if stream:
            async for chunk in response.aiter_text():
                print(chunk, end="", flush=True)
            return {"status": "stream complete"}
        return response.json()

    async def generate_beam(
        self,
        state_id: str,
        num_beams: int = 5,
        max_length: int = 50,
        diversity_penalty: Optional[float] = None,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """Generate text using beam search."""
        data = {
            "state_id": state_id,
            "num_beams": num_beams,
            "max_length": max_length,
            "diversity_penalty": diversity_penalty,
            "early_stopping": early_stopping
        }
        response = await self._make_request("POST", "/generate/beam", json=data)
        return response.json()

    async def analyze_attention(
        self,
        state_id: str,
        layer: Optional[int] = None,
        threshold: float = 0.1,
        head: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze attention patterns."""
        data = {
            "state_id": state_id,
            "layer": layer,
            "threshold": threshold,
            "head": head
        }
        response = await self._make_request("POST", f"/analysis/attention/{state_id}", json=data)
        return response.json()

    async def inspect_cache(
        self,
        state_id: str,
        key_name: Optional[str] = None,
        layer_range: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Inspect key/value cache."""
        data = {
            "state_id": state_id,
            "key_name": key_name,
            "layer_range": layer_range
        }
        response = await self._make_request("POST", f"/analysis/cache/{state_id}", json=data)
        return response.json()

async def run_demo():
    """Run a comprehensive demo of the Cacher API."""
    async with CacherClient() as client:
        try:
            # 1. Create a state
            logger.info("1. Creating a new state...")
            state = await client.create_state(
                "gpt2",
                "Once upon a time in a digital realm",
                config={"temperature": 0.7, "top_p": 0.9}
            )
            state_id = state["state_id"]
            logger.info(f"Created state with ID: {state_id}")

            # 2. Generate tokens with streaming
            logger.info("\n2. Generating tokens with streaming...")
            await client.generate_tokens(
                state_id,
                max_tokens=20,
                temperature=0.8,
                stream=True
            )

            # 3. Generate with beam search
            logger.info("\n3. Generating with beam search...")
            beam_results = await client.generate_beam(
                state_id,
                num_beams=3,
                max_length=30,
                diversity_penalty=0.5
            )
            logger.info("Beam search results:")
            for i, beam in enumerate(beam_results["beams"], 1):
                logger.info(f"Beam {i}: {beam['text']} (score: {beam['score']:.3f})")

            # 4. Analyze attention patterns
            logger.info("\n4. Analyzing attention patterns...")
            attention_data = await client.analyze_attention(
                state_id,
                layer=5,
                threshold=0.1
            )
            logger.info("Attention analysis:")
            logger.info(json.dumps(attention_data, indent=2))

            # 5. Inspect cache
            logger.info("\n5. Inspecting cache...")
            cache_data = await client.inspect_cache(
                state_id,
                layer_range=[0, 1]
            )
            logger.info("Cache inspection:")
            logger.info(json.dumps(cache_data, indent=2))

            # 6. Clean up
            logger.info("\n6. Cleaning up...")
            await client.delete_state(state_id)
            logger.info("State deleted successfully")

        except CacherAPIError as e:
            logger.error(f"API Error: {e.message}")
            if e.details:
                logger.error(f"Details: {e.details}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_demo())