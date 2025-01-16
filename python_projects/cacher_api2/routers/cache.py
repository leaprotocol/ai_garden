from fastapi import APIRouter, HTTPException, Depends
from cacher_api2.services.cache_service import CacheService
from cacher_api2.schemas import *
from cacher_api2.utils import get_logger
from typing import Annotated
from cacher_api2.dependencies import get_cache_service

router = APIRouter()
logger = get_logger(__name__)

@router.post("/save_cache/{model_id}/{cache_id}/{filename}", response_model=SaveCacheResponse)
async def save_cache(
    model_id: str,
    cache_id: str,
    filename: str,
    cache_service: Annotated[CacheService, Depends(get_cache_service)]
):
    """
    Saves a cache to a file.
    """
    logger.debug(f"POST /save_cache/{model_id}/{cache_id} called with filename: {filename}")
    try:
        filepath = cache_service.save_cache(model_id, cache_id, filename)
        return SaveCacheResponse(success=True, filepath=filepath)
    except ValueError as e:
        logger.error(f"ValueError in save_cache: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in save_cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/load_cache/{model_id}/{filename}", response_model=LoadCacheResponse)
async def load_cache(
    model_id: str,
    filename: str,
    cache_service: Annotated[CacheService, Depends(get_cache_service)]
):
    """
    Loads a cache from a file.
    """
    logger.debug(f"POST /load_cache/{model_id}/{filename} called with filename: {filename}")
    try:
        cache_id = cache_service.load_cache(model_id, filename)
        return LoadCacheResponse(cache_id=cache_id)
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in load_cache: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in load_cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/list_caches/{model_id}", response_model=ListCachesResponse)
async def list_caches(
    model_id: str,
    cache_service: Annotated[CacheService, Depends(get_cache_service)]
):
    """
    Lists all saved caches for a given model and their sizes.
    """
    logger.debug(f"GET /list_caches/{model_id} called")
    try:
        caches = cache_service.list_caches(model_id)
        return ListCachesResponse(caches=caches)
    except Exception as e:
        logger.error(f"Unexpected error in list_caches: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/cache_size/{model_id}/{cache_id}", response_model=CacheSizeResponse)
async def get_cache_size(
    model_id: str,
    cache_id: str,
    cache_service: Annotated[CacheService, Depends(get_cache_service)]
):
    """
    Gets the size of a specific cache.
    """
    logger.debug(f"GET /cache_size/{model_id}/{cache_id} called")
    try:
        cache_size = cache_service.get_cache_size(cache_id)
        return CacheSizeResponse(cache_size=cache_size)
    except ValueError as e:
        logger.error(f"ValueError in get_cache_size: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_cache_size: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 