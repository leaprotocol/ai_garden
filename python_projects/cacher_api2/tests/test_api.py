import pytest
from fastapi.testclient import TestClient
from cacher_api2.main import app
import os

# Test forward pass
def test_forward_pass(test_client):
    model_id = "SmolLM"
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    assert response.status_code == 200
    assert "cache_id" in response.json()
    assert "input_length" in response.json()

# Test saving and loading cache
def test_save_and_load_cache(test_client, tmp_path_factory):
    model_id = "SmolLM"
    # 1. Do a forward pass to create a cache
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    cache_id = response.json()["cache_id"]

    # 2. Save the cache
    filename = "test_cache.pt"
    response = test_client.post(f"/cache/save_cache/{model_id}/{cache_id}?filename={filename}")
    assert response.status_code == 200
    
    # Assert that the cache file exists in the default saved_caches directory
    expected_cache_path = os.path.join("saved_caches", model_id, filename)
    assert os.path.exists(expected_cache_path)

    # 3. Load the cache
    response = test_client.post(f"/cache/load_cache/{model_id}/{filename}")
    assert response.status_code == 200
    loaded_cache_id = response.json()["cache_id"]
    assert loaded_cache_id is not None
    assert loaded_cache_id != cache_id # Ensure a new cache ID is generated on load

# Test getting next token probabilities
def test_get_next_token_probs(test_client):
    model_id = "SmolLM"
    # 1. Do a forward pass to create a cache
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    cache_id = response.json()["cache_id"]

    # 2. Get next token probabilities
    response = test_client.get(f"/generation/next_token_probs/{model_id}/{cache_id}")
    assert response.status_code == 200
    assert isinstance(response.json()["probabilities"], dict)

# Test beam search
def test_beam_search(test_client):
    model_id = "SmolLM"
    # 1. Do a forward pass to create a cache
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    cache_id = response.json()["cache_id"]

    # 2. Perform beam search
    response = test_client.get(f"/generation/beam_search/{model_id}/{cache_id}?beam_width=3&max_length=10")
    assert response.status_code == 200
    assert "sequences" in response.json()

# Test getting a greedy token
def test_get_greedy_token(test_client):
    model_id = "SmolLM"
    # 1. Do a forward pass to create a cache
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    cache_id = response.json()["cache_id"]

    # 2. Get the greedy token
    response = test_client.get(f"/generation/greedy_token/{model_id}/{cache_id}")
    assert response.status_code == 200
    assert "token" in response.json()
    assert "probability" in response.json()

# Test listing caches
def test_list_caches(test_client):
    model_id = "SmolLM"
    # Create some caches
    response1 = test_client.post(f"/generation/forward/{model_id}", json={"text": "Test cache 1"})
    cache_id1 = response1.json()["cache_id"]
    test_client.post(f"/cache/save_cache/{model_id}/{cache_id1}?filename=cache1.pt")

    response2 = test_client.post(f"/generation/forward/{model_id}", json={"text": "Test cache 2"})
    cache_id2 = response2.json()["cache_id"]
    test_client.post(f"/cache/save_cache/{model_id}/{cache_id2}?filename=cache2.pt")

    # List caches
    response = test_client.get(f"/cache/list_caches/{model_id}")
    assert response.status_code == 200
    caches = response.json()["caches"]
    assert isinstance(caches, dict)
    assert len(caches) >= 2

# Test getting cache size
def test_get_cache_size(test_client):
    model_id = "SmolLM"
    # 1. Do a forward pass to create a cache
    response = test_client.post(f"/generation/forward/{model_id}", json={"text": "The capital of France is"})
    cache_id = response.json()["cache_id"]

    # 2. Get cache size
    response = test_client.get(f"/cache/cache_size/{model_id}/{cache_id}")
    assert response.status_code == 200
    assert "cache_size" in response.json()
    assert response.json()["cache_size"] > 0

# Test invalid cache ID
def test_invalid_cache_id(test_client):
    model_id = "SmolLM"
    response = test_client.get(f"/generation/next_token_probs/{model_id}/invalid_cache_id")
    assert response.status_code == 400

# Test saving cache with invalid ID
def test_save_cache_invalid_id(test_client):
    model_id = "SmolLM"
    response = test_client.post(f"/cache/save_cache/{model_id}/invalid_cache_id?filename=test.pt")
    assert response.status_code == 400

# Test loading non-existent cache file
def test_load_nonexistent_cache(test_client):
    model_id = "SmolLM"
    response = test_client.post(f"/cache/load_cache/{model_id}/nonexistent_cache.pt")
    assert response.status_code == 404 