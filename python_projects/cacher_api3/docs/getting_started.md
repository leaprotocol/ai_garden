# Getting Started with Cacher API v3

This guide will help you get up and running with the Cacher API v3.

## Installation

```bash
pip install cacher-api3
```

## Basic Usage

1. **Start the API server:**

   ```bash
   uvicorn cacher_api3.main:app --reload
   ```

2. **Create a new state:**

   ```http
   POST /v1/states
   Content-Type: application/json

   {
       "model_id": "gpt2",
       "text": "Hello, world!",
       "model_type": "causal_lm"
   }
   ```

3. **Generate text:**

   ```http
   POST /v1/generate/token
   Content-Type: application/json

   {
       "state_id": "state-abc123",
       "max_tokens": 5
   }
   ```

## Environment Variables

| Variable                     | Description                                      | Default Value |
| ---------------------------- | ------------------------------------------------ | ------------- |
| `CACHER_MODEL_NAME`          | Default model name to use                        | gpt2          |
| `CACHER_DEVICE`              | Device to use for model inference (cpu or cuda) | cpu           |
| `CACHER_DEFAULT_TEMPERATURE` | Default temperature for generation               | 0.7           |
| `CACHER_DEFAULT_TOP_P`       | Default top_p for generation                     | 0.9           |
| `CACHER_DEFAULT_SEED`        | Default seed for reproducibility                 | 42            |
| `CACHER_DEFAULT_CONTEXT_SIZE`| Default context window size                      | 2048          |
| `CACHER_LOGITS_STORAGE`      | Default logits storage mode                      | top_k         |
| `CACHER_TOP_K_LOGITS`        | Default number of top logits to store            | 20            |
