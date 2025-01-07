import pytest
from fastapi.testclient import TestClient
from cacher_api2.main import app
from cacher_api2.models import ModelManager
from cacher_api2.services.cache_service import CacheService
from cacher_api2.services.generation_service import GenerationService
from cacher_api2.dependencies import get_cache_service, get_generation_service
import os
from cacher_api2.utils import get_logger
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = get_logger(__name__)

@pytest.fixture(scope="session")
def model_manager():
    """
    Provides a ModelManager instance for testing.
    """
    return ModelManager()

@pytest.fixture(scope="session")
def cache_service():
    """
    Provides a CacheService instance for testing.
    """
    return CacheService()

@pytest.fixture(scope="session")
def generation_service(model_manager, cache_service):
    """
    Provides a GenerationService instance for testing.
    """
    return GenerationService(model_manager, cache_service)

@pytest.fixture(scope="session")
def test_client(generation_service, cache_service):
    """
    Provides a test client for the FastAPI application.
    """
    app.dependency_overrides[get_generation_service] = lambda: generation_service
    app.dependency_overrides[get_cache_service] = lambda: cache_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

@pytest.fixture(scope="session", autouse=True)
def setup_environment(tmp_path_factory):
    """
    Sets up the environment before tests are run.
    """
    # Set environment variables for testing
    os.environ["MODEL_NAME"] = "HuggingFaceTB/SmolLM2-360M-Instruct"
    os.environ["DEVICE"] = "cpu"
    os.environ["LOAD_IN_8BIT"] = "False"

    # Create a temporary directory for caches
    cache_dir = tmp_path_factory.mktemp("caches")
    os.environ["CACHE_DIR"] = str(cache_dir)
    logger.info(f"Environment variables set: MODEL_NAME={os.environ['MODEL_NAME']}, DEVICE={os.environ['DEVICE']}, LOAD_IN_8BIT={os.environ['LOAD_IN_8BIT']}, CACHE_DIR={os.environ['CACHE_DIR']}")

    yield

    # Clean up environment variables after tests are run
    del os.environ["MODEL_NAME"]
    del os.environ["DEVICE"]
    del os.environ["LOAD_IN_8BIT"]
    del os.environ["CACHE_DIR"] 