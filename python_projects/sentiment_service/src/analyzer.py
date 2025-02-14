import json
import logging
import asyncio
import torch
import websockets
from typing import Union, List, Dict
from rich.logging import RichHandler
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("sentiment_service")

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", device: str = None):
        self.model_name = model_name
        self.pipeline = None
        # Auto-detect device if not specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        
    async def initialize(self):
        """Initialize the sentiment analysis pipeline asynchronously."""
        log.info(f"Loading sentiment analysis model: {self.model_name}")
        try:
            # Run in executor to not block the event loop
            loop = asyncio.get_event_loop()
            self.pipeline = await loop.run_in_executor(
                None, 
                lambda: pipeline("sentiment-analysis", model=self.model_name, device=self.device)
            )
            log.info(f"Model loaded successfully on {self.device}")
            if self.device == "cuda":
                log.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        except Exception as e:
            log.error(f"Failed to load model: {str(e)}")
            raise
    
    async def analyze_text(self, text: str) -> Dict:
        """Analyze a single text."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized")
            
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.pipeline(text)[0]
        )
        
        # Map score to custom sentiment labels
        score = result["score"]
        if result["label"] == "POSITIVE":
            sentiment = "VERY_BULLISH" if score > 0.75 else "BULLISH"
        else:
            sentiment = "VERY_BEARISH" if score > 0.75 else "BEARISH"
            score = 1 - score  # Invert score for bearish sentiments
            
        return {
            "sentiment_score": (2 * score - 1),  # Convert to [-1, 1] range
            "confidence": score,
            "sentiment": sentiment,
            "explanation": f"Analysis based on {self.model_name} model with {score:.2f} confidence"
        }
    
    async def analyze_texts(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in batch."""
        return [await self.analyze_text(text) for text in texts]

async def handle_client(websocket, analyzer: SentimentAnalyzer):
    """Handle individual client connections."""
    client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"New client connected from {client_info}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                log.debug(f"Received request from {client_info}: {data}")
                
                if "text" in data:
                    result = await analyzer.analyze_text(data["text"])
                elif "texts" in data:
                    result = await analyzer.analyze_texts(data["texts"])
                else:
                    result = {"error": "Invalid request format"}
                    
                await websocket.send(json.dumps(result))
                log.debug(f"Sent response to {client_info}: {result}")
                
            except json.JSONDecodeError:
                error_msg = {"error": "Invalid JSON"}
                await websocket.send(json.dumps(error_msg))
                log.warning(f"Invalid JSON received from {client_info}")
            except Exception as e:
                log.error(f"Error processing request from {client_info}: {str(e)}")
                await websocket.send(json.dumps({"error": str(e)}))
                
    except websockets.exceptions.ConnectionClosed:
        log.info(f"Client {client_info} disconnected")
    except Exception as e:
        log.error(f"Unexpected error with client {client_info}: {str(e)}")

async def main(host: str = "0.0.0.0", port: int = 8080):
    """Main server function."""
    try:
        analyzer = SentimentAnalyzer()
        await analyzer.initialize()
        
        async with websockets.serve(
            lambda ws: handle_client(ws, analyzer),
            host,
            port
        ):
            log.info(f"Server running at ws://{host}:{port}")
            log.info(f"For local connections use: ws://localhost:{port}")
            await asyncio.Future()  # run forever
    except Exception as e:
        log.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server shutting down")
    except Exception as e:
        log.error(f"Fatal error: {str(e)}")
        raise
