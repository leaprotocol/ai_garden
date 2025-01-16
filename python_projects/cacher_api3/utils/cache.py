import logging
import time
from pathlib import Path
from typing import List
from ..config import settings

logger = logging.getLogger(__name__)

def cleanup_cache(cache_dir: Path = settings.CACHE_DIR, max_age: int = 86400) -> List[Path]:
    """Clean up old cache files.
    
    Args:
        cache_dir: Directory containing cache files
        max_age: Maximum age of cache files in seconds (default: 24 hours)
        
    Returns:
        List of removed file paths
    """
    if not cache_dir.exists():
        logger.warning(f"Cache directory {cache_dir} does not exist")
        return []
    
    current_time = time.time()
    removed_files = []
    
    for cache_file in cache_dir.glob("*.json"):
        try:
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > max_age:
                cache_file.unlink()
                removed_files.append(cache_file)
                logger.debug(f"Removed old cache file: {cache_file}")
        except Exception as e:
            logger.error(f"Error cleaning up cache file {cache_file}: {str(e)}")
    
    if removed_files:
        logger.info(f"Cleaned up {len(removed_files)} old cache files")
    
    return removed_files

def get_cache_size(cache_dir: Path = settings.CACHE_DIR) -> int:
    """Get the current number of cache files.
    
    Args:
        cache_dir: Directory containing cache files
        
    Returns:
        Number of cache files
    """
    if not cache_dir.exists():
        return 0
    
    return len(list(cache_dir.glob("*.json")))

def get_cache_stats(cache_dir: Path = settings.CACHE_DIR) -> dict:
    """Get cache statistics.
    
    Args:
        cache_dir: Directory containing cache files
        
    Returns:
        Dictionary with cache statistics
    """
    if not cache_dir.exists():
        return {
            "total_files": 0,
            "total_size": 0,
            "oldest_file": None,
            "newest_file": None
        }
    
    cache_files = list(cache_dir.glob("*.json"))
    if not cache_files:
        return {
            "total_files": 0,
            "total_size": 0,
            "oldest_file": None,
            "newest_file": None
        }
    
    total_size = sum(f.stat().st_size for f in cache_files)
    file_times = [(f, f.stat().st_mtime) for f in cache_files]
    oldest = min(file_times, key=lambda x: x[1])
    newest = max(file_times, key=lambda x: x[1])
    
    return {
        "total_files": len(cache_files),
        "total_size": total_size,
        "oldest_file": {
            "name": oldest[0].name,
            "age": time.time() - oldest[1]
        },
        "newest_file": {
            "name": newest[0].name,
            "age": time.time() - newest[1]
        }
    } 