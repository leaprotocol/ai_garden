# Infinite Tokens

A simple project that generates and streams tokens infinitely using language models. This tool continuously generates text, showing a real-time stream of tokens being created by the model.

## Features

- Infinite text generation from any Hugging Face model
- Real-time streaming of generated tokens
- Configurable generation parameters
- Automatic device selection (CUDA, MPS, or CPU)
- Detailed logging

## Requirements

This project uses the dependencies defined in the parent directory's `pyproject.toml`:
- Python 3.9+
- PyTorch
- Transformers
- Rich (for console output)

## Installation

The project is designed to work with the existing poetry environment:

```bash
# Navigate to the parent directory where pyproject.toml is located
cd python_projects

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# OR
# .\venv\Scripts\activate  # On Windows

# Run the program
python -m infinite_tokens.main
```

## Usage

Basic usage:

```bash
python -m infinite_tokens.main
```

With custom parameters:

```bash
python -m infinite_tokens.main --model "gpt2" --prompt "In a world where" --temperature 0.8 --chunk-size 30 --delay 0.02
```

### Command-line Arguments

- `--model`: Model name from Hugging Face (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
- `--prompt`: Initial text to start generation (default: "Once upon a time")
- `--temperature`: Sampling temperature - higher values produce more random outputs (default: 0.7)
- `--chunk-size`: Number of tokens to generate in each chunk (default: 20)
- `--delay`: Delay between token displays in seconds (default: 0.05)
- `--device`: Device to use - "cuda", "mps", or "cpu" (default: auto-detect)

## Examples

### Basic example
```bash
python -m infinite_tokens.main
```

### Using a specific model with custom settings
```bash
python -m infinite_tokens.main --model "facebook/opt-350m" --prompt "The future of AI looks" --temperature 0.9
```

### Generating quickly with minimal delay
```bash
python -m infinite_tokens.main --delay 0.01 --chunk-size 50
```

## Stopping Generation

Press `Ctrl+C` to stop the generation process. The program will display the total number of tokens generated.

## Logs

Logs are stored in the `logs/infinite_tokens.log` file for debugging and tracking purposes. 