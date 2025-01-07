import requests
import json
import logging
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# API Base URL (Update if your API runs on a different port/host)
API_BASE_URL = "http://localhost:8000"  # Assuming default FastAPI port

@dataclass
class CacheInfo:
    """Represents information about a cached item."""
    cache_id: str
    filename: str
    size: int

def handle_error(response: requests.Response):
    """Handles API errors."""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"API Error: {e}")
        try:
            error_details = response.json()
            logger.error(f"Error Details: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"Raw Response: {response.text}")
        sys.exit(1)

def save_cache(model_id: str, cache_id: str, filename: str) -> str:
    """Saves a cache to a file."""
    url = f"{API_BASE_URL}/save_cache/{model_id}/{cache_id}"
    payload = {"filename": filename}
    response = requests.post(url, json=payload)
    handle_error(response)
    filepath = response.json()["filepath"]
    logger.info(f"Cache saved to: {filepath}")
    return filepath

def load_cache(model_id: str, filename: str) -> str:
    """Loads a cache from a file."""
    url = f"{API_BASE_URL}/load_cache/{model_id}/{filename}"
    response = requests.post(url)
    handle_error(response)
    cache_id = response.json()["cache_id"]
    logger.info(f"Cache loaded: {cache_id}")
    return cache_id

def list_caches(model_id: str) -> List[CacheInfo]:
    """Lists all saved caches for a given model."""
    url = f"{API_BASE_URL}/list_caches/{model_id}"
    response = requests.get(url)
    handle_error(response)
    caches_data = response.json()["caches"]
    caches = [CacheInfo(**data) for data in caches_data]
    logger.info(f"Caches for model {model_id}:")
    for cache in caches:
        logger.info(f"  - ID: {cache.cache_id}, Filename: {cache.filename}, Size: {cache.size} bytes")
    return caches

def get_cache_size(model_id: str, cache_id: str) -> int:
    """Gets the size of a specific cache."""
    url = f"{API_BASE_URL}/cache_size/{model_id}/{cache_id}"
    response = requests.get(url)
    handle_error(response)
    cache_size = response.json()["cache_size"]
    logger.info(f"Cache size for {cache_id}: {cache_size} bytes")
    return cache_size

def main():
    """Demonstrates the caching API functionality."""
    logger.info("Starting Caching API Demo")

    model_id = "my_model"  # Replace with your model ID
    cache_id = "example_cache"  # Replace with your desired cache ID
    filename = "example_cache.pickle"  # Replace with your desired filename

    # Demonstrate saving a cache
    logger.info("Saving a cache...")
    filepath = save_cache(model_id, cache_id, filename)

    # Demonstrate listing caches
    logger.info("Listing caches...")
    list_caches(model_id)

    # Demonstrate getting cache size
    logger.info("Getting cache size...")
    get_cache_size(model_id, cache_id)

    # Demonstrate loading a cache
    logger.info("Loading a cache...")
    loaded_cache_id = load_cache(model_id, filename)

    # Verify loaded cache ID
    if loaded_cache_id == cache_id:
        logger.info("Cache loaded successfully!")
    else:
        logger.error("Loaded cache ID does not match original cache ID!")

    logger.info("Caching API Demo Completed")

if __name__ == "__main__":
    main()