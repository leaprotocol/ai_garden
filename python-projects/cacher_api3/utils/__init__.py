"""
Utility functions for Cacher API v3
"""

from .logging import setup_logging
from .cache import cleanup_cache

__all__ = ["setup_logging", "cleanup_cache"] 