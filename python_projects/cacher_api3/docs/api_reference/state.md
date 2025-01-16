# State Management Endpoints

## Create State

*Request:*

```http
POST /v1/states
Content-Type: application/json

{
    "model_id": "gpt2",
    "text": "Optional initial text",
    "model_type": "causal_lm", // Optional: "causal_lm", "sequence_classification", or "auto" (default)
    "config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "context_size": 2048,
        "store_logits": "top_k",
        "top_k_logits": 20
    }
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "model_id": "gpt2",
    "head_position": 0,
    "text": "Optional initial text",
    "model_type": "causal_lm",
    "config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "context_size": 2048,
        "store_logits": "top_k",
        "top_k_logits": 20
    },
    "created_at": "2024-04-28T10:00:00Z"
}
```

**Note:** The `model_type` field is optional. If not provided, the API will attempt to automatically detect the model type.

## Get State

*Request:*

```http
GET /v1/states/{state_id}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "model_id": "gpt2",
    "head_position": 10,
    "text": "The quick brown fox",
    "model_type": "causal_lm",
    "config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "context_size": 2048,
        "store_logits": "top_k",
        "top_k_logits": 20
    },
    "created_at": "2024-04-28T10:00:00Z",
    "modified_at": "2024-04-28T10:01:00Z"
}
```

## Delete State

*Request:*

```http
DELETE /v1/states/{state_id}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "message": "State deleted successfully"
}
```

## Update State Configuration

*Request:*

```http
POST /v1/states/config
Content-Type: application/json

{
    "state_id": "state-abc123",
    "config": {
        "temperature": 0.8,
        "seed": null
    }
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "config": {
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": null,
        "context_size": 2048,
        "store_logits": "top_k",
        "top_k_logits": 20
    }
}
```
