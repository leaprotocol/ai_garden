#!/bin/bash

# --- Script to create documentation for cacher_api3 ---

# --- Create directories ---
mkdir -p cacher_api3/docs/api_reference
mkdir -p cacher_api3/docs/core_concepts

# --- Create docs/index.md ---
cat << EOF > cacher_api3/docs/index.md
# Cacher API v3 Documentation

Welcome to the official documentation for the Cacher API v3! This guide provides comprehensive information about the API's features, usage, and development.

## Table of Contents

- [Getting Started](getting_started.md)
- [API Reference](api_reference/index.md)
  - [State Management](api_reference/state.md)
  - [Generation](api_reference/generation.md)
  - [Analysis](api_reference/analysis.md)
  - [Model Management](api_reference/models.md)
- [Core Concepts](core_concepts/index.md)
  - [States](core_concepts/states.md)
  - [Operations](core_concepts/operations.md)
- [Advanced Usage](advanced_usage.md)
- [Development](development.md)
- [Error Handling](error_handling.md)
- [FAQ](faq.md)

## Core Principles

- **Statefulness:** Maintain conversational context and model state across interactions.
- **Reproducibility:** Ensure deterministic outputs through controllable seeding.
- **Flexibility:** Support various generation strategies and analysis techniques.
- **Efficiency:** Optimize for performance and resource utilization.
- **Extensibility:** Design a modular architecture for future enhancements.
EOF

# --- Create docs/getting_started.md ---
cat << EOF > cacher_api3/docs/getting_started.md
# Getting Started with Cacher API v3

This guide will help you get up and running with the Cacher API v3.

## Installation

\`\`\`bash
pip install cacher-api3
\`\`\`

## Basic Usage

1. **Start the API server:**

   \`\`\`bash
   uvicorn cacher_api3.main:app --reload
   \`\`\`

2. **Create a new state:**

   \`\`\`http
   POST /v1/states
   Content-Type: application/json

   {
       "model_id": "gpt2",
       "text": "Hello, world!",
       "model_type": "causal_lm"
   }
   \`\`\`

3. **Generate text:**

   \`\`\`http
   POST /v1/generate/token
   Content-Type: application/json

   {
       "state_id": "state-abc123",
       "max_tokens": 5
   }
   \`\`\`

## Environment Variables

| Variable                     | Description                                      | Default Value |
| ---------------------------- | ------------------------------------------------ | ------------- |
| \`CACHER_MODEL_NAME\`          | Default model name to use                        | gpt2          |
| \`CACHER_DEVICE\`              | Device to use for model inference (cpu or cuda) | cpu           |
| \`CACHER_DEFAULT_TEMPERATURE\` | Default temperature for generation               | 0.7           |
| \`CACHER_DEFAULT_TOP_P\`       | Default top_p for generation                     | 0.9           |
| \`CACHER_DEFAULT_SEED\`        | Default seed for reproducibility                 | 42            |
| \`CACHER_DEFAULT_CONTEXT_SIZE\`| Default context window size                      | 2048          |
| \`CACHER_LOGITS_STORAGE\`      | Default logits storage mode                      | top_k         |
| \`CACHER_TOP_K_LOGITS\`        | Default number of top logits to store            | 20            |
EOF

# --- Create docs/api_reference/index.md ---
cat << EOF > cacher_api3/docs/api_reference/index.md
# API Reference

The Cacher API v3 is organized around resources (states, models, etc.) and actions (generate, forward, analyze).

## Endpoints

- [State Management](state.md)
- [Generation](generation.md)
- [Analysis](analysis.md)
- [Model Management](models.md)
EOF

# --- Create docs/api_reference/state.md ---
cat << EOF > cacher_api3/docs/api_reference/state.md
# State Management Endpoints

## Create State

*Request:*

\`\`\`http
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
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

**Note:** The \`model_type\` field is optional. If not provided, the API will attempt to automatically detect the model type.

## Get State

*Request:*

\`\`\`http
GET /v1/states/{state_id}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Delete State

*Request:*

\`\`\`http
DELETE /v1/states/{state_id}
\`\`\`

*Response:*

\`\`\`json
{
    "state_id": "state-abc123",
    "message": "State deleted successfully"
}
\`\`\`

## Update State Configuration

*Request:*

\`\`\`http
POST /v1/states/config
Content-Type: application/json

{
    "state_id": "state-abc123",
    "config": {
        "temperature": 0.8,
        "seed": null
    }
}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`
EOF

# --- Create docs/api_reference/generation.md ---
cat << EOF > cacher_api3/docs/api_reference/generation.md
# Generation Endpoints

## Generate Tokens

*Request:*

\`\`\`http
POST /v1/generate/token
Content-Type: application/json

{
    "state_id": "state-abc123",
    "max_tokens": 5,
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": null,
    "stop_sequences": ["\\n", "."],
    "stream": false
}
\`\`\`

*Response:*

\`\`\`json
{
    "state_id": "state-abc123",
    "text": " jumps over the lazy",
    "new_tokens": [" jumps", " over", " the", " lazy", " dog"],
    "probabilities": [0.8, 0.7, 0.9, 0.6, 0.95],
    "finished": false,
    "finish_reason": null
}
\`\`\`

**Note:** When \`seed\` is set to \`null\`, the generation will be non-deterministic.

## Generate Multiple Continuations

*Request:*

\`\`\`http
POST /v1/generate/multi
Content-Type: application/json

{
    "state_id": "state-abc123",
    "n": 3,
    "max_tokens": 10,
    "temperature": 0.7
}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Beam Search Generation

*Request:*

\`\`\`http
POST /v1/generate/beam
Content-Type: application/json

{
    "state_id": "state-abc123",
    "num_beams": 5,
    "max_length": 20,
    "early_stopping": true
}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`
EOF

# --- Create docs/api_reference/analysis.md ---
cat << EOF > cacher_api3/docs/api_reference/analysis.md
# Analysis Endpoints

## Analyze Tokens

*Request:*

\`\`\`http
GET /v1/analyze/tokens/{state_id}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Analyze Attention

*Request:*

\`\`\`http
POST /v1/analyze/attention
Content-Type: application/json

{
    "state_id": "state-abc123",
    "layer": 0,
    "head": 0
}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Inspect Cache

*Request:*

\`\`\`http
POST /v1/analyze/cache/inspect
Content-Type: application/json

{
    "state_id": "state-abc123",
    "layer": 0,
    "key_name": "k"
}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Classify

*Request:*

\`\`\`http
POST /v1/analyze/classify
Content-Type: application/json

{
    "state_id": "state-reward-model",
    "text": "This is a positive sentence."
}
\`\`\`

*Response:*

\`\`\`json
{
    "state_id": "state-reward-model",
    "label": "positive",
    "score": 0.95
}
\`\`\`

**Note:** This endpoint is only applicable to states created with sequence classification models.

## Get Embedding

*Request:*

\`\`\`http
POST /v1/analyze/embedding
Content-Type: application/json

{
    "state_id": "state-abc123",
    "token_index": 5
}
\`\`\`

*Response:*

\`\`\`json
{
    "state_id": "state-abc123",
    "token_index": 5,
    "embedding": [
        0.1, 0.2, 0.3
    ]
}
\`\`\`
EOF

# --- Create docs/api_reference/models.md ---
cat << EOF > cacher_api3/docs/api_reference/models.md
# Model Management Endpoints

## List Models

*Request:*

\`\`\`http
GET /v1/models
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`

## Get Model

*Request:*

\`\`\`http
GET /v1/models/{model_id}
\`\`\`

*Response:*

\`\`\`json
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
\`\`\`
EOF

# --- Create docs/core_concepts/index.md ---
cat << EOF > cacher_api3/docs/core_concepts/index.md
# Core Concepts

## [States](states.md)

A state represents the internal memory of a language model at a specific point in time.

## [Operations](operations.md)

Operations are actions performed on or with states.
EOF

# --- Create docs/core_concepts/states.md ---
cat << EOF > cacher_api3/docs/core_concepts/states.md
# States

A state represents the internal memory of a language model at a specific point in time. It encapsulates:

- **Model Configuration:** Parameters like temperature, top_p, and seed.
- **Context History:** The sequence of tokens processed so far.
- **Attention Mask:** Information about which tokens the model should attend to.
- **Cache:** Key-value pairs used for efficient generation.
- **Metadata:** Information about the state's creation and modifications.

States are central to the Cacher API, allowing for:

- **Context Management:** Continuing conversations and maintaining context over multiple turns.
- **Reproducibility:** Generating the same output given the same state and parameters.
- **Forking and Merging:** Creating variations of a state and combining them.
- **Analysis:** Inspecting the internal workings of the model at a specific state.
EOF

# --- Create docs/core_concepts/operations.md ---
cat << EOF > cacher_api3/docs/core_concepts/operations.md
# Operations

Operations are actions performed on or with states. Key operations include:

- **Creation:** Initializing a new state with a model and initial text.
- **Generation:** Extending a state by generating new tokens.
- **Forward:** Processing text through a state without generating new tokens.
- **Analysis:** Inspecting the state's internal data, such as attention patterns and token probabilities.
- **Modification:** Updating a state's configuration or history (planned).
- **Forking:** Creating a copy of a state.
- **Merging:** Combining two or more states (planned).
EOF

# --- Create docs/advanced_usage.md ---
cat << EOF > cacher_api3/docs/advanced_usage.md
# Advanced Usage

## Configuration

The Cacher API can be configured using environment variables. See the [Getting Started](getting_started.md) guide for a list of available variables.

## Customization

(Details about extending the API, implementing custom model loaders, etc.)
EOF

# --- Create docs/development.md ---
cat << EOF > cacher_api3/docs/development.md
# Development

## Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## Development Setup

1. **Install development dependencies:**

   \`\`\`bash
   poetry install --with dev
   \`\`\`

2. **Run tests:**

   \`\`\`bash
   poetry run pytest
   \`\`\`

3. **Run with debug logging:**

   \`\`\`bash
   CACHER_DEBUG=True poetry run uvicorn cacher_api3.main:app --reload
   \`\`\`

## Project Structure

\`\`\`
cacher_api3/
├── api/                 # FastAPI application
│   ├── routes/          # API endpoints
│   │   ├── __init__.py
│   │   ├── analysis.py  # Analysis routes
│   │   ├── generation.py# Generation routes
│   │   └── state.py     # State management routes
│   └── main.py          # FastAPI app creation
├── core/                # Core logic
│   ├── __init__.py
│   ├── model_loader.py  # Model loading utilities
│   ├── state_manager.py # State management
│   └── generation_utils.py # Generation utilities
├── config.py            # Configuration settings
├── schemas.py           # Pydantic models
├── tests/               # Unit and integration tests
├── .env.example         # Example environment variables
├── main.py              # FastAPI application
├── demo.py              # Usage examples
└── demo_new_features.py # New features demo
\`\`\`
EOF

# --- Create docs/error_handling.md ---
cat << EOF > cacher_api3/docs/error_handling.md
# Error Handling

The Cacher API uses standard HTTP status codes:

- \`200\`: Success
- \`400\`: Bad Request (e.g., invalid input)
- \`404\`: Not Found (e.g., state not found)
- \`500\`: Internal Server Error

Errors include detailed messages and are logged for debugging.
EOF

# --- Create docs/faq.md ---
cat << EOF > cacher_api3/docs/faq.md
# Frequently Asked Questions

(Add common questions and answers here)
EOF

echo "Documentation created successfully in cacher_api3/docs/"
