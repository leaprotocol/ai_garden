import pytest
from fastapi.testclient import TestClient
from ..main import app
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_state(client):
    """Create a test state and return its ID."""
    response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Test text"}
    )
    assert response.status_code == 200
    return response.json()["state_id"]

@pytest.fixture
def cleanup_cache():
    """Clean up the cache directory after tests."""
    yield
    cache_dir = Path("cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Set up logging for tests."""
    import logging
    caplog.set_level(logging.INFO)
    yield 