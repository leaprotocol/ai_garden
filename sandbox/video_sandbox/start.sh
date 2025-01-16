#!/bin/bash

# Activate virtual environment if using
# source venv/bin/activate

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000