import pytest
from fastapi.testclient import TestClient
from cacher_api2.main import app
from cacher_api2.services.cache_service import CacheService
from cacher_api2.services.generation_service import GenerationService

# Create a test client
client = TestClient(app)

# Fixture for creating a temporary cache directory
@pytest.fixture(scope="module")
def temp_cache_dir(tmp_path_factory):
    dir = tmp_path_factory.mktemp("caches")
    return str(dir)

# Fixture for the API instance
@pytest.fixture(scope="module")
def api_instance(temp_cache_dir):
    cache_service = CacheService(cache_dir=temp_cache_dir)
    generation_service = GenerationService(app.state.model_manager, cache_service)
    app.state.cache_service = cache_service
    app.state.generation_service = generation_service
    return app

@pytest.mark.benchmark(group="forward")
def test_forward_pass_no_cache_benchmark(api_instance, benchmark):
    model_id = "SmolLM"
    def forward_pass():
        response = client.post(f"/generation/forward/{model_id}", json={"text": "The quick brown fox jumps over the lazy dog"})
        assert response.status_code == 200
    benchmark(forward_pass)

@pytest.mark.benchmark(group="forward")
def test_forward_pass_with_cache_benchmark(api_instance, benchmark):
    model_id = "SmolLM"
    # Create a cache first
    response = client.post(f"/generation/forward/{model_id}", json={"text": "The quick brown fox jumps over the lazy dog"})
    cache_id = response.json()["cache_id"]

    def forward_pass():
        response = client.post(f"/generation/forward/{model_id}", json={"text": "And then", "cache_id": cache_id})
        assert response.status_code == 200
    benchmark(forward_pass)

@pytest.mark.benchmark(group="generation")
def test_generate_greedy_token_benchmark(api_instance, benchmark):
    model_id = "SmolLM"
    # Create a cache first
    response = client.post(f"/generation/forward/{model_id}", json={"text": "The quick brown fox jumps over the lazy dog"})
    cache_id = response.json()["cache_id"]

    def generate_token():
        response = client.get(f"/generation/greedy_token/{model_id}/{cache_id}")
        assert response.status_code == 200
    benchmark(generate_token)

@pytest.mark.benchmark(group="cache")
def test_save_cache_benchmark(api_instance, benchmark):
    model_id = "SmolLM"
    # Create a cache first
    response = client.post(f"/generation/forward/{model_id}", json={"text": "The quick brown fox jumps over the lazy dog"})
    cache_id = response.json()["cache_id"]

    def save_cache():
        response = client.post(f"/cache/save_cache/{model_id}/{cache_id}?filename=benchmark_cache.pt")
        assert response.status_code == 200
    benchmark(save_cache)

@pytest.mark.benchmark(group="cache")
def test_load_cache_benchmark(api_instance, benchmark):
    model_id = "SmolLM"
    # Create and save a cache first
    response = client.post(f"/generation/forward/{model_id}", json={"text": "The quick brown fox jumps over the lazy dog"})
    cache_id = response.json()["cache_id"]
    client.post(f"/cache/save_cache/{model_id}/{cache_id}?filename=benchmark_cache.pt")

    def load_cache():
        response = client.post(f"/cache/load_cache/{model_id}/benchmark_cache.pt")
        assert response.status_code == 200
    benchmark(load_cache) 