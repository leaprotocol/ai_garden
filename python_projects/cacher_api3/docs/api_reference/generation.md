# Generation Endpoints

## Generate Tokens

*Request:*

```http
POST /v1/generate/token
Content-Type: application/json

{
    "state_id": "state-abc123",
    "max_tokens": 5,
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": null,
    "stop_sequences": ["\n", "."],
    "stream": false
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "text": " jumps over the lazy",
    "new_tokens": [" jumps", " over", " the", " lazy", " dog"],
    "probabilities": [0.8, 0.7, 0.9, 0.6, 0.95],
    "finished": false,
    "finish_reason": null
}
```

**Note:** When `seed` is set to `null`, the generation will be non-deterministic.

## Generate Multiple Continuations

*Request:*

```http
POST /v1/generate/multi
Content-Type: application/json

{
    "state_id": "state-abc123",
    "n": 3,
    "max_tokens": 10,
    "temperature": 0.7
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "continuations": [
        {
            "text": " jumps over the lazy dog.",
            "new_tokens": [" jumps", " over", " the", " lazy", " dog", "."],
            "probabilities": [0.8, 0.7, 0.9, 0.6, 0.95, 0.99],
            "finished": true,
            "finish_reason": "stop_sequence"
        },
        {
            "text": " jumps over the lazy fox.",
            "new_tokens": [" jumps", " over", " the", " lazy", " fox", "."],
            "probabilities": [0.8, 0.7, 0.9, 0.6, 0.90, 0.98],
            "finished": true,
            "finish_reason": "stop_sequence"
        },
        {
            "text": " jumps over the lazy cat and",
            "new_tokens": [" jumps", " over", " the", " lazy", " cat", " and"],
            "probabilities": [0.8, 0.7, 0.9, 0.6, 0.85, 0.7],
            "finished": false,
            "finish_reason": null
        }
    ]
}
```

## Beam Search Generation

*Request:*

```http
POST /v1/generate/beam
Content-Type: application/json

{
    "state_id": "state-abc123",
    "num_beams": 5,
    "max_length": 20,
    "early_stopping": true
}
```

*Response:*

```json
{
    "state_id": "state-abc123",
    "beams": [
        " jumps over the lazy dog and continues running.",
        " jumps over the lazy dog and then stops.",
        " jumps over the lazy dog and looks around.",
        " jumps over the lazy dog and starts barking.",
        " jumps over the lazy dog and wags its tail."
    ]
}
```
