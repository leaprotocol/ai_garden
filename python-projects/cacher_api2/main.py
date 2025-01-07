from fastapi import FastAPI
from cacher_api2.routers import generation, cache
from cacher_api2.models import ModelManager
from cacher_api2.services.cache_service import CacheService
from cacher_api2.services.generation_service import GenerationService
from cacher_api2.utils import get_logger
from contextlib import asynccontextmanager

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes and shuts down the application, including loading models and setting up services.
    """
    logger.info("Starting up...")
    
    # Initialize services and managers
    app.state.model_manager = ModelManager()
    app.state.cache_service = CacheService()
    app.state.generation_service = GenerationService(app.state.model_manager, app.state.cache_service)
    
    logger.info("Startup complete")
    yield
    
    logger.info("Shutting down...")
    # Perform any necessary cleanup here
    logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

# Include routers
app.include_router(generation.router, prefix="/generation", tags=["generation"])
app.include_router(cache.router, prefix="/cache", tags=["cache"])

@app.get("/")
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Cacher API!"} 