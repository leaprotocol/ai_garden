import os
import torch
from cacher_api2.utils import get_logger, generate_cache_id
from typing import Dict, Optional

logger = get_logger(__name__)

class CacheService:
    def __init__(self, cache_dir: str = "saved_caches"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cached_states: Dict[str, Dict] = {}

    def save_cache(self, model_id: str, cache_id: str, filename: str) -> str:
        logger.info(f"Saving cache for model {model_id} with cache_id {cache_id} to {filename}")
        if cache_id not in self.cached_states:
            raise ValueError("Cache not found")

        cache = self.cached_states[cache_id]
        filepath = os.path.join(self.cache_dir, model_id, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        try:
            torch.save(cache, filepath)
            logger.info(f"Cache saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            raise

    def load_cache(self, model_id: str, filename: str) -> str:
        logger.info(f"Loading cache for model {model_id} from {filename}")
        filepath = os.path.join(self.cache_dir, model_id, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("Cache file not found")

        try:
            loaded_cache = torch.load(filepath)
            logger.info(f"Cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            raise

        cache_id = generate_cache_id()
        self.cached_states[cache_id] = loaded_cache
        return cache_id

    def list_caches(self, model_id: str) -> Dict[str, int]:
        logger.info(f"Listing caches for model {model_id}")
        model_cache_dir = os.path.join(self.cache_dir, model_id)

        if not os.path.exists(model_cache_dir):
            return {}

        cache_files = {}
        for filename in os.listdir(model_cache_dir):
            filepath = os.path.join(model_cache_dir, filename)
            if os.path.isfile(filepath):
                cache_files[filename] = os.path.getsize(filepath)
        return cache_files

    def get_cache_size(self, cache_id: str) -> int:
        logger.info(f"Getting size for cache {cache_id}")
        if cache_id not in self.cached_states:
            raise ValueError("Cache not found")

        cache = self.cached_states[cache_id]
        cache_size = 0
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                cache_size += value.element_size() * value.nelement()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        cache_size += v.element_size() * v.nelement()
        return cache_size

    def add_to_cache(self, cache_id: str, cache_data: Dict):
        """Adds a new cache entry or updates an existing one."""
        self.cached_states[cache_id] = cache_data 