import pytest
from fastapi.testclient import TestClient
from ..main import app
import json

client = TestClient(app)

def test_create_state():
    """Test creating a new state."""
    response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Hello, world!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "state_id" in data
    assert "metadata" in data

def test_get_state():
    """Test retrieving a state."""
    # First create a state
    create_response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Hello, world!"}
    )
    state_id = create_response.json()["state_id"]
    
    # Then retrieve it
    response = client.get(f"/states/{state_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["state_id"] == state_id

def test_generate_token():
    """Test token generation."""
    # Create a state first
    create_response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Hello"}
    )
    state_id = create_response.json()["state_id"]
    
    # Generate tokens
    response = client.post(
        "/generate/token",
        json={
            "state_id": state_id,
            "max_tokens": 1,
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "state_id" in data

def test_analyze_attention():
    """Test attention pattern analysis."""
    # Create a state first
    create_response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Hello, world!"}
    )
    state_id = create_response.json()["state_id"]
    
    # Analyze attention
    response = client.post(
        f"/analysis/attention/{state_id}",
        json={"state_id": state_id, "layer": 0}
    )
    assert response.status_code == 200
    data = response.json()
    assert "attention_patterns" in data
    assert len(data["attention_patterns"]) > 0

def test_invalid_state_id():
    """Test handling of invalid state IDs."""
    response = client.get("/states/nonexistent-id")
    assert response.status_code == 404

def test_stream_generation():
    """Test streaming token generation."""
    # Create a state first
    create_response = client.post(
        "/states",
        json={"model_id": "test-model", "text": "Hello"}
    )
    state_id = create_response.json()["state_id"]
    
    # Generate tokens with streaming
    with client.stream(
        "POST",
        "/generate/token",
        json={
            "state_id": state_id,
            "max_tokens": 5,
            "stream": True
        }
    ) as response:
        assert response.status_code == 200
        # Check that we get some chunks
        chunks = [chunk for chunk in response.iter_text()]
        assert len(chunks) > 0 