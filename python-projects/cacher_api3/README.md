# Cacher API v3

A stateful caching API for language models that provides efficient state management, generation, and analysis capabilities. Built with FastAPI and modern Python async features.

## Features

### State Management
- Persistent state storage with JSON files
- Automatic UUID-based state identification
- State forking and merging capabilities
- Configurable state cleanup and cache management
- Metadata tracking (creation time, modifications)

### Generation
- Token-by-token streaming generation
- Configurable parameters (temperature, top_p)
- Beam search with diversity penalties
- Stop sequence support
- Real-time streaming responses
- Configurable batch sizes

### Analysis & Introspection
- Token-level analysis with logprobs
- Attention pattern visualization with thresholding
- Key/value cache inspection by layer
- Normalized attention patterns
- Layer and head-specific analysis

## Quick Start

1. Install dependencies using Poetry:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

2. Start the server:
```bash
poetry run uvicorn cacher_api3.main:app --reload
```

3. Run the demo:
```bash
poetry run python -m cacher_api3.demo
```

## Client Usage

```python
import asyncio
from cacher_api3.demo import CacherClient

async def main():
    async with CacherClient() as client:
        # Create a state
        state = await client.create_state(
            "gpt2",
            "Once upon a time",
            config={"temperature": 0.7, "top_p": 0.9}
        )
        state_id = state["state_id"]

        # Generate with streaming
        await client.generate_tokens(
            state_id,
            max_tokens=10,
            temperature=0.8,
            stream=True
        )

        # Generate with beam search
        beams = await client.generate_beam(
            state_id,
            num_beams=3,
            max_length=30,
            diversity_penalty=0.5
        )

        # Analyze attention patterns
        attention = await client.analyze_attention(
            state_id,
            layer=5,
            threshold=0.1
        )

        # Clean up
        await client.delete_state(state_id)

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### State Management

#### Create State
```http
POST /states
Content-Type: application/json

{
    "model_id": "gpt2",
    "text": "Optional initial text",
    "config": {
        "temperature": 0.7,
        "top_p": 0.9
    }
}
```

#### Get State
```http
GET /states/{state_id}
```

#### Delete State
```http
DELETE /states/{state_id}
```

### Generation

#### Generate Tokens
```http
POST /generate/token
Content-Type: application/json

{
    "state_id": "...",
    "max_tokens": 10,
    "temperature": 0.8,
    "top_p": 0.9,
    "stop_sequences": [".", "\n"],
    "stream": true
}
```

#### Beam Search
```http
POST /generate/beam
Content-Type: application/json

{
    "state_id": "...",
    "num_beams": 5,
    "max_length": 50,
    "diversity_penalty": 0.5,
    "early_stopping": true
}
```

### Analysis

#### Token Analysis
```http
GET /analysis/tokens/{state_id}
```

#### Attention Analysis
```http
POST /analysis/attention/{state_id}
Content-Type: application/json

{
    "layer": 5,
    "threshold": 0.1,
    "head": 0
}
```

#### Cache Inspection
```http
POST /analysis/cache/{state_id}
Content-Type: application/json

{
    "key_name": "optional_key",
    "layer_range": [0, 5]
}
```

## Configuration

Configuration is handled through environment variables with the prefix `CACHER_`. Available settings:

```bash
# API Settings
CACHER_DEBUG=True
CACHER_LOG_LEVEL=DEBUG

# Cache Settings
CACHER_CACHE_DIR=./cache
CACHER_MAX_CACHE_SIZE=1000
CACHER_CACHE_CLEANUP_INTERVAL=3600

# Model Settings
CACHER_DEFAULT_MODEL=gpt2

# Security Settings
CACHER_ENABLE_CORS=True
CACHER_API_KEY_REQUIRED=False
```

See `config.py` for all available settings.

## Project Structure
```
cacher_api3/
├── api/
│   ├── routes/
│   │   ├── state.py      # State management endpoints
│   │   ├── generation.py # Generation endpoints
│   │   └── analysis.py   # Analysis endpoints
│   └── models/
│       └── schemas.py    # Pydantic models
├── core/
│   ├── state_manager.py  # State management logic
│   └── generation.py     # Generation logic
├── utils/
│   ├── cache.py         # Cache utilities
│   └── logging.py       # Logging setup
├── config.py            # Configuration management
├── main.py             # FastAPI application
└── demo.py             # Usage examples
```

## Development

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Run tests:
```bash
poetry run pytest
```

3. Run with debug logging:
```bash
CACHER_DEBUG=True poetry run uvicorn cacher_api3.main:app --reload
```

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 404: State not found
- 500: Server error

Errors include detailed messages and are logged for debugging.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.