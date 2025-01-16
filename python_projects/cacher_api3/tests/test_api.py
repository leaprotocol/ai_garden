import pytest
from fastapi.testclient import TestClient
from ..main import app
import json
from httpx import AsyncClient

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

async def test_forward_new_state(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "state_id" in data
    assert data["head_position"] == 1

async def test_forward_existing_state(async_client: AsyncClient):
    # Create a state first
    create_response = await async_client.post("/v1/states", json={"model_id": "gpt2", "text": "Hello"})
    assert create_response.status_code == 200
    state_id = create_response.json()["state_id"]

    response = await async_client.post("/v1/forward", json={"state_id": state_id, "text_input": " world"})
    assert response.status_code == 200
    data = response.json()
    assert data["state_id"] == state_id
    assert data["head_position"] == 2

async def test_forward_return_final_token_info(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello", "return_final_token_info": True})
    assert response.status_code == 200
    data = response.json()
    assert "token_info" in data
    assert data["token_info"]["token"] == "Hello"

async def test_forward_return_all_tokens_info(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello world", "return_all_tokens_info": True})
    assert response.status_code == 200
    data = response.json()
    assert "tokens_info" in data
    assert len(data["tokens_info"]) == 2
    assert data["tokens_info"][0]["token"] == "Hello"
    assert data["tokens_info"][1]["token"] == " world"

async def test_forward_stream_tokens_info(async_client: AsyncClient):
    async with async_client.stream("POST", "/v1/forward", json={"text_input": "Hello world", "stream_tokens_info": True}) as response:
        assert response.status_code == 200
        count = 0
        async for line in response.aiter_lines():
            assert "token_info" in line
            count += 1
        assert count == 2

async def test_forward_with_logits(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello", "return_logits": True})
    assert response.status_code == 200
    data = response.json()
    assert "token_info" in data
    assert "logits" in data["token_info"]

async def test_forward_with_top_logprobs(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello", "return_top_logprobs": 2})
    assert response.status_code == 200
    data = response.json()
    assert "token_info" in data
    assert "top_logprobs" in data["token_info"]
    assert len(data["token_info"]["top_logprobs"]) == 2

async def test_forward_with_entropy(async_client: AsyncClient):
    response = await async_client.post("/v1/forward", json={"text_input": "Hello", "return_entropy": True})
    assert response.status_code == 200
    data = response.json()
    assert "token_info" in data
    assert "entropy" in data["token_info"]

async def test_generate_new_state(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "The quick brown fox", "max_new_tokens": 5})
    assert response.status_code == 200
    data = response.json()
    assert "state_id" in data
    assert "generated_text" in data
    assert data["head_position"] > 4 # Initial 4 tokens + generated

async def test_generate_existing_state(async_client: AsyncClient):
    # Create a state first
    create_response = await async_client.post("/v1/states", json={"model_id": "gpt2", "text": "The quick brown fox"})
    assert create_response.status_code == 200
    state_id = create_response.json()["state_id"]

    response = await async_client.post("/v1/generate", json={"state_id": state_id, "max_new_tokens": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["state_id"] == state_id
    assert "generated_text" in data
    assert data["head_position"] > 4

async def test_generate_return_final_token_info(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "Hello", "max_new_tokens": 1, "return_final_token_info": True})
    assert response.status_code == 200
    data = response.json()
    assert "generated_token_info" in data
    assert "token" in data["generated_token_info"]

async def test_generate_return_all_tokens_info(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "Hello", "max_new_tokens": 2, "return_all_tokens_info": True})
    assert response.status_code == 200
    data = response.json()
    assert "generated_tokens_info" in data
    assert len(data["generated_tokens_info"]) == 2
    assert "token" in data["generated_tokens_info"][0]

async def test_generate_stream_tokens_info(async_client: AsyncClient):
    async with async_client.stream("POST", "/v1/generate", json={"text_input": "Hello", "max_new_tokens": 2, "stream_tokens_info": True}) as response:
        assert response.status_code == 200
        count = 0
        async for line in response.aiter_lines():
            assert "token_info" in line
            count += 1
        assert count == 2

async def test_generate_with_logits(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "Hello", "max_new_tokens": 1, "return_logits": True})
    assert response.status_code == 200
    data = response.json()
    assert "generated_token_info" in data
    assert "logits" in data["generated_token_info"]

async def test_generate_with_top_logprobs(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "Hello", "max_new_tokens": 1, "return_top_logprobs": 2})
    assert response.status_code == 200
    data = response.json()
    assert "generated_token_info" in data
    assert "top_logprobs" in data["generated_token_info"]
    assert len(data["generated_token_info"]["top_logprobs"]) == 2

async def test_generate_with_entropy(async_client: AsyncClient):
    response = await async_client.post("/v1/generate", json={"text_input": "Hello", "max_new_tokens": 1, "return_entropy": True})
    assert response.status_code == 200
    data = response.json()
    assert "generated_token_info" in data
    assert "entropy" in data["generated_token_info"]

async def test_tokenize(async_client: AsyncClient):
    response = await async_client.post("/v1/tokenize", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) == 2

async def test_detokenize(async_client: AsyncClient):
    response = await async_client.post("/v1/detokenize", json={"tokens": [15496, 1157]})
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["text"] == "Hello world" 