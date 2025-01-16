# Model Management Endpoints

## List Models

*Request:*

```http
GET /v1/models
```

*Response:*

```json
[
    {
        "model_id": "gpt2",
        "description": "GPT-2 language model"
    },
    {
        "model_id": "bert-base-uncased",
        "description": "BERT base model"
    }
]
```

## Get Model

*Request:*

```http
GET /v1/models/{model_id}
```

*Response:*

```json
{
    "model_id": "gpt2",
    "description": "GPT-2 language model",
    "config": {
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": null,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        "scale_attn_weights": true,
        "use_cache": true,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "model_type": "gpt2"
    }
}
```
