# Cacher API v3

**Stateful, Reproducible, and Efficient API for Language Model Operations**

[![Tests](https://github.com/username/cacher_api3/actions/workflows/tests.yml/badge.svg)](https://github.com/username/cacher_api3/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/cacher-api3/badge/?version=latest)](https://cacher-api3.readthedocs.io/en/latest/?badge=latest)

## Overview

Cacher API v3 is a powerful and flexible API designed for interacting with language models, emphasizing statefulness, reproducibility, and efficiency. It provides a robust framework for managing conversational context, controlling generation parameters, and analyzing model behavior.

## Key Features

- **Stateful Operations:** Maintain context and model state across multiple interactions, enabling coherent and context-aware conversations.
- **Reproducibility:** Ensure deterministic outputs through controllable seeding and configuration management.
- **Flexible Generation:** Support for various generation strategies, including token-by-token generation, multiple continuations, and beam search.
- **In-depth Analysis:** Tools for inspecting token probabilities, attention patterns, and internal cache data.
- **Modular Design:** Easily extensible architecture for integrating new models and functionalities.
- **Efficient Performance:** Optimized for fast response times and minimal resource consumption.

## Documentation

**Comprehensive documentation is available in the [`docs`](cacher_api3/docs) directory.** It covers:

- **[Getting Started](cacher_api3/docs/getting_started.md):** Installation, setup, and basic usage.
- **[API Reference](cacher_api3/docs/api_reference/index.md):** Detailed descriptions of all API endpoints.
- **[Core Concepts](cacher_api3/docs/core_concepts/index.md):** Explanation of fundamental concepts like states and operations.
- **[Advanced Usage](cacher_api3/docs/advanced_usage.md):** Guidance on configuration, customization, and advanced features.
- **[Development](cacher_api3/docs/development.md):** Information for contributors and developers.
- **[Error Handling](cacher_api3/docs/error_handling.md):** Error codes and troubleshooting tips.
- **[FAQ](cacher_api3/docs/faq.md):** Frequently asked questions.

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

## Contributing

Contributions are welcome! Please see the [Development](cacher_api3/docs/development.md) section in the documentation for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.