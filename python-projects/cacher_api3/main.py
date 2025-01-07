import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description="A stateful caching API for language models",
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Configure CORS if enabled
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Import routers after app creation to avoid circular imports
from .api.routes import state, generation, analysis

# Register routers
app.include_router(state.router, prefix="/states", tags=["State Management"])
app.include_router(generation.router, prefix="/generate", tags=["Generation"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    # Initialize services and resources here
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    # Cleanup resources here
    logger.info("Shutdown complete")

@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint works!"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.DEBUG
    ) 