# Development

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

1. **Install development dependencies:**

   ```bash
   poetry install --with dev
   ```

2. **Run tests:**

   ```bash
   poetry run pytest
   ```

3. **Run with debug logging:**

   ```bash
   CACHER_DEBUG=True poetry run uvicorn cacher_api3.main:app --reload
   ```

## Project Structure

```
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
```
