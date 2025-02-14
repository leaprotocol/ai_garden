# Sentiment Analysis Service

A simple WebSocket-based service that provides real-time sentiment analysis for text, optimized for financial/market sentiment analysis.

## Features

- WebSocket server for real-time analysis
- Support for both single text and batch analysis
- Sentiment scores mapped to market sentiment (VERY_BULLISH, BULLISH, BEARISH, VERY_BEARISH)
- Confidence scores and explanations
- Asynchronous processing
- Rich logging

## Setup

1. Make sure you have Python 3.9+ installed
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install dependencies:
   ```bash
   cd sentiment_service
   poetry install
   ```

## Running the Service

```bash
poetry run python src/analyzer.py
```

The service will start on `ws://localhost:8765` by default.

## Usage Examples

### Python Client Example

```python
import asyncio
import websockets
import json

async def analyze_text(text):
    async with websockets.connect('ws://localhost:8765') as websocket:
        await websocket.send(json.dumps({"text": text}))
        return json.loads(await websocket.recv())

async def analyze_batch(texts):
    async with websockets.connect('ws://localhost:8765') as websocket:
        await websocket.send(json.dumps({"texts": texts}))
        return json.loads(await websocket.recv())

# Example usage
async def main():
    # Single text analysis
    result = await analyze_text("Bitcoin is showing strong momentum!")
    print("Single analysis:", result)
    
    # Batch analysis
    results = await analyze_batch([
        "Market sentiment is very negative",
        "New partnership announced with major tech company"
    ])
    print("Batch analysis:", results)

if __name__ == "__main__":
    asyncio.run(main())
```

## API Format

### Request Format

Single text:
```json
{
    "text": "Bitcoin is showing strong momentum!"
}
```

Batch mode:
```json
{
    "texts": [
        "Bitcoin price drops 5%",
        "New major partnership announced"
    ]
}
```

### Response Format

Single text:
```json
{
    "sentiment_score": 0.8,        // Range: -1 to 1
    "confidence": 0.9,             // Range: 0 to 1
    "sentiment": "VERY_BULLISH",   // VERY_BULLISH, BULLISH, BEARISH, VERY_BEARISH
    "explanation": "Analysis based on distilbert model with 0.90 confidence"
}
```

## Notes

- The service uses the `distilbert-base-uncased-finetuned-sst-2-english` model by default
- Sentiment scores are mapped from the model's binary classification to a -1 to 1 range
- The service handles disconnections and errors gracefully
- Rich logging provides clear visibility into the service's operation
