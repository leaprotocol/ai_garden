from fastapi import Request
from cacher_api2.services.cache_service import CacheService
from cacher_api2.services.generation_service import GenerationService

def get_cache_service(request: Request) -> CacheService:
    """
    Dependency that provides the CacheService instance.
    """
    return request.app.state.cache_service

def get_generation_service(request: Request) -> GenerationService:
    """
    Dependency that provides the GenerationService instance.
    """ 