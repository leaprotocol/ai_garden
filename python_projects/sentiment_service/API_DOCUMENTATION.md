# Sentiment Analysis Service API Documentation

## Overview
The Sentiment Analysis Service provides real-time sentiment analysis for financial and market-related text through a WebSocket API. It's designed for integration with algorithmic trading systems, offering both single-text and batch analysis capabilities.

## Quick Start

### Connection Details
- **WebSocket URL**: `ws://localhost:8080`
- **Remote URL**: `ws://184.144.229.106:8080` (when deployed)
- **Protocol**: WebSocket (WSS)

### Python Example
```python
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_client")

async def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Sentiment analysis result
    """
    uri = "ws://localhost:8080"
    try:
        async with websockets.connect(uri) as websocket:
            request = {"text": text}
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            return json.loads(response)
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise

async def analyze_batch(texts: list[str]) -> list[dict]:
    """
    Analyze sentiment of multiple texts in batch.
    
    Args:
        texts (list[str]): List of texts to analyze
        
    Returns:
        list[dict]: List of sentiment analysis results
    """
    uri = "ws://localhost:8080"
    try:
        async with websockets.connect(uri) as websocket:
            request = {"texts": texts}
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            return json.loads(response)
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {str(e)}")
        raise

# Example usage
async def main():
    # Single analysis
    result = await analyze_sentiment("Bitcoin breaks above key resistance level")
    logger.info(f"Single analysis result: {result}")
    
    # Batch analysis
    results = await analyze_batch([
        "Market shows signs of weakness",
        "Strong institutional buying pressure"
    ])
    logger.info(f"Batch analysis results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

## API Endpoints

### 1. Single Text Analysis
Analyzes sentiment for a single piece of text.

#### Request Format
```json
{
    "text": "Bitcoin is showing strong momentum!"
}
```

#### Response Format
```json
{
    "sentiment_score": 0.8,        // Range: -1 to 1
    "confidence": 0.9,             // Range: 0 to 1
    "sentiment": "VERY_BULLISH",   // Enum: VERY_BULLISH, BULLISH, BEARISH, VERY_BEARISH
    "explanation": "Analysis based on distilbert model with 0.90 confidence"
}
```

### 2. Batch Text Analysis
Analyzes sentiment for multiple texts in a single request.

#### Request Format
```json
{
    "texts": [
        "Bitcoin price drops 5%",
        "New major partnership announced"
    ]
}
```

#### Response Format
```json
[
    {
        "sentiment_score": -0.6,
        "confidence": 0.8,
        "sentiment": "BEARISH",
        "explanation": "Analysis based on distilbert model with 0.80 confidence"
    },
    {
        "sentiment_score": 0.9,
        "confidence": 0.95,
        "sentiment": "VERY_BULLISH",
        "explanation": "Analysis based on distilbert model with 0.95 confidence"
    }
]
```

## Sentiment Categories

The service maps sentiment scores to the following categories:

| Category | Score Range | Description |
|----------|-------------|-------------|
| VERY_BULLISH | 0.75 to 1.0 | Strong positive sentiment |
| BULLISH | 0.0 to 0.75 | Moderate positive sentiment |
| BEARISH | -0.75 to 0.0 | Moderate negative sentiment |
| VERY_BEARISH | -1.0 to -0.75 | Strong negative sentiment |

## Error Handling

### Common Error Responses
```json
{
    "error": "Invalid JSON"
}
```
```json
{
    "error": "Invalid request format"
}
```

### Python Error Handling Example
```python
async def analyze_with_retry(text: str, max_retries: int = 3) -> dict:
    """
    Analyze sentiment with retry logic.
    
    Args:
        text (str): Text to analyze
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: Sentiment analysis result
        
    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return await analyze_sentiment(text)
        except websockets.exceptions.ConnectionClosed:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
```

## Best Practices

1. **Connection Management**
   - Reuse WebSocket connections for multiple requests when possible
   - Implement proper connection cleanup
   - Handle reconnection logic for production use

2. **Rate Limiting**
   - Implement rate limiting in your client
   - Recommended: max 100 requests per second per client

3. **Batch Processing**
   - Use batch requests for multiple texts when possible
   - Optimal batch size: 10-20 texts per request

4. **Error Handling**
   - Implement retry logic with exponential backoff
   - Handle connection errors gracefully
   - Log errors and responses for debugging

## Production Usage

### Connection String Format
```python
# Local development
WS_URL = "ws://localhost:8080"

# Production (with SSL)
WS_URL = "wss://your-domain:8080"  # If using SSL
```

### Health Check Example
```python
async def check_service_health() -> bool:
    """Check if the sentiment service is healthy."""
    try:
        result = await analyze_sentiment("test message")
        return "sentiment_score" in result
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False
```

### Monitoring Example
```python
async def monitor_service_performance():
    """Monitor service performance metrics."""
    start_time = time.time()
    result = await analyze_sentiment("test message")
    latency = time.time() - start_time
    
    logger.info(f"Request latency: {latency:.3f}s")
    logger.info(f"Response confidence: {result['confidence']}")
```

## Model Information

- Model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Task: Sentiment Analysis
- Input: English text (max 512 tokens)
- Output: Sentiment classification with confidence scores
- Hardware: Running on CUDA-enabled GPU for optimal performance

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if the service is running
   - Verify port forwarding is active
   - Ensure no firewall blocking

2. **Slow Response Times**
   - Check batch size (reduce if too large)
   - Monitor server GPU usage
   - Verify network latency

3. **Invalid Results**
   - Ensure text is in English
   - Check for text length limits
   - Verify JSON format

### Debugging Tools

1. **Service Logs**
   ```bash
   # View service logs
   ssh -p 45314 root@184.144.229.106 "tail -f /root/sentiment_service/service.log"
   ```

2. **Connection Test**
   ```bash
   # Test WebSocket connection
   websocat ws://localhost:8080
   ```

## Support

For issues or questions:
1. Check service logs for errors
2. Monitor GPU usage with `nvidia-smi`
3. Use the provided monitoring tools in tmux
4. Contact support with relevant logs and error messages 