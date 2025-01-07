import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Cacher API v3"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Cache Settings
    CACHE_DIR: Path = Path("cache")
    MAX_CACHE_SIZE: int = 1000  # Maximum number of states to keep in memory
    CACHE_CLEANUP_INTERVAL: int = 3600  # Cleanup interval in seconds
    
    # Model Settings
    DEFAULT_MODEL: str = "gpt2"
    MODEL_CONFIG: Dict[str, Any] = {
        "gpt2": {
            "max_length": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    # Generation Settings
    MAX_TOKENS_PER_REQUEST: int = 100
    STREAM_CHUNK_SIZE: int = 1
    
    # Analysis Settings
    MAX_ATTENTION_HEADS: int = 12
    ATTENTION_THRESHOLD: float = 0.1
    
    # Security Settings
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: list = ["*"]
    API_KEY_REQUIRED: bool = False
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_prefix = "CACHER_"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure cache directory exists
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Environment-specific settings
if settings.DEBUG:
    settings.LOG_LEVEL = "DEBUG" 