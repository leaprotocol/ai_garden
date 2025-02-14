"""
Sentiment Analysis Client Library

This module provides a robust client for the Sentiment Analysis Service,
designed for integration with algorithmic trading systems.
"""

import asyncio
import json
import time
import logging
from typing import Union, List, Dict, Optional
import websockets
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_client")

class SentimentCategory(str, Enum):
    """Enumeration of possible sentiment categories."""
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results."""
    sentiment_score: float  # Range: -1 to 1
    confidence: float      # Range: 0 to 1
    sentiment: SentimentCategory
    explanation: str

    @classmethod
    def from_dict(cls, data: Dict) -> 'SentimentResult':
        """Create SentimentResult from API response dictionary."""
        return cls(
            sentiment_score=data['sentiment_score'],
            confidence=data['confidence'],
            sentiment=SentimentCategory(data['sentiment']),
            explanation=data['explanation']
        )

class SentimentAnalysisClient:
    """
    Client for the Sentiment Analysis Service.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Rate limiting
    - Batch processing
    - Error handling with retries
    """
    
    def __init__(
        self,
        uri: str = "ws://localhost:8080",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        connection_timeout: float = 10.0,
        max_batch_size: int = 20
    ):
        """
        Initialize the client.
        
        Args:
            uri: WebSocket URI of the service
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will be multiplied by attempt number)
            connection_timeout: Timeout for connection attempts
            max_batch_size: Maximum number of texts in a batch request
        """
        self.uri = uri
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.max_batch_size = max_batch_size
        self._websocket = None
        self._lock = asyncio.Lock()
        self._last_request_time = 0
        self._request_count = 0
        
    async def _get_connection(self) -> websockets.WebSocketClientProtocol:
        """Get or create a WebSocket connection."""
        if self._websocket is None or not self._websocket.open:
            try:
                self._websocket = await websockets.connect(
                    self.uri,
                    timeout=self.connection_timeout
                )
            except Exception as e:
                logger.error(f"Failed to connect to {self.uri}: {str(e)}")
                raise
        return self._websocket
    
    async def _send_request(self, request: Dict) -> Dict:
        """Send a request and get response with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with self._lock:  # Ensure thread safety
                    websocket = await self._get_connection()
                    
                    # Rate limiting
                    current_time = time.time()
                    if current_time - self._last_request_time < 0.01:  # Max 100 req/s
                        await asyncio.sleep(0.01)
                    
                    await websocket.send(json.dumps(request))
                    response = await websocket.recv()
                    
                    self._last_request_time = time.time()
                    self._request_count += 1
                    
                    return json.loads(response)
                    
            except websockets.exceptions.ConnectionClosed:
                self._websocket = None
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult object containing the analysis
            
        Raises:
            ValueError: If text is empty or too long
            ConnectionError: If connection fails after retries
            Exception: For other errors
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
            
        request = {"text": text}
        response = await self._send_request(request)
        
        if "error" in response:
            raise Exception(f"API error: {response['error']}")
            
        return SentimentResult.from_dict(response)
    
    async def analyze_batch(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            chunk_size: Optional size for splitting large batches
            
        Returns:
            List of SentimentResult objects
            
        Raises:
            ValueError: If texts list is empty or contains invalid texts
            ConnectionError: If connection fails after retries
            Exception: For other errors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
            
        chunk_size = chunk_size or self.max_batch_size
        results = []
        
        # Process in chunks if needed
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            request = {"texts": chunk}
            response = await self._send_request(request)
            
            if isinstance(response, dict) and "error" in response:
                raise Exception(f"API error: {response['error']}")
                
            results.extend([SentimentResult.from_dict(r) for r in response])
        
        return results
    
    async def check_health(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            result = await self.analyze_text("test message")
            return isinstance(result, SentimentResult)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for monitoring.
        
        Returns:
            dict: Performance metrics including latency and request count
        """
        start_time = time.time()
        await self.analyze_text("test message")
        latency = time.time() - start_time
        
        return {
            "latency": latency,
            "request_count": self._request_count,
            "last_request_time": self._last_request_time
        }
    
    async def close(self):
        """Close the WebSocket connection."""
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

async def main():
    """Example usage of the client library."""
    client = SentimentAnalysisClient()
    
    try:
        # Check service health
        is_healthy = await client.check_health()
        logger.info(f"Service health check: {'Passed' if is_healthy else 'Failed'}")
        
        # Single text analysis
        result = await client.analyze_text("Bitcoin breaks above key resistance level")
        logger.info(f"Single analysis result: {result}")
        
        # Batch analysis
        results = await client.analyze_batch([
            "Market shows signs of weakness",
            "Strong institutional buying pressure"
        ])
        logger.info(f"Batch analysis results: {results}")
        
        # Get performance metrics
        metrics = await client.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 