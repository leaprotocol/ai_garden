# Analysis Endpoints

## Analyze Tokens

*Request:*

```http
GET /v1/analyze/tokens/{state_id}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "tokens": [
        {
            "text": "The",
            "position": 0,
            "logprob": -0.123
        },
        {
            "text": " quick",
            "position": 1,
            "logprob": -0.456
        }
    ]
}
```

## Analyze Attention

*Request:*

```http
POST /v1/analyze/attention
Content-Type: application/json

{
    "state_id": "state-abc123",
    "layer": 0,
    "head": 0
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "layer": 0,
    "head": 0,
    "attention_pattern": [
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1],
        [0.3, 0.6, 0.1]
    ]
}
```

## Inspect Cache

*Request:*

```http
POST /v1/analyze/cache/inspect
Content-Type: application/json

{
    "state_id": "state-abc123",
    "layer": 0,
    "key_name": "k"
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "layer": 0,
    "key_name": "k",
    "cache_data": [
        [...],
        [...],
        [...]
    ]
}
```

## Classify

*Request:*

```http
POST /v1/analyze/classify
Content-Type: application/json

{
    "state_id": "state-reward-model",
    "text": "This is a positive sentence."
}
```

*Response:*

```json
{
    "state_id": "state-reward-model",
    "label": "positive",
    "score": 0.95
}
```

**Note:** This endpoint is only applicable to states created with sequence classification models.

## Get Embedding

*Request:*

```http
POST /v1/analyze/embedding
Content-Type: application/json

{
    "state_id": "state-abc123",
    "token_index": 5
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "token_index": 5,
    "embedding": [
        0.1, 0.2, 0.3
    ]
}
```
