"""
WebSocket server for streaming tokens to web clients.
"""

import json
import asyncio
import logging
from typing import Optional, Set, List
import websockets
from websockets.server import WebSocketServerProtocol, WebSocketServer
from token_generator import TokenGenerator

logger = logging.getLogger(__name__)

class TokenWebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.generator = TokenGenerator()
        self.current_generation_task: Optional[asyncio.Task] = None
        self.server: Optional[WebSocketServer] = None
        self.tasks: List[asyncio.Task] = []
        
        # Add callback for token events
        self.generator.add_callback(self.broadcast_token)
        
    def broadcast_token(self, event_data: dict) -> None:
        """Broadcast token data to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps(event_data)
        websockets.broadcast(self.clients, message)
        
    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection"""
        try:
            self.clients.add(websocket)
            logger.debug(f"Client connected. Total clients: {len(self.clients)}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    
                    if command == "generate":
                        # Stop any existing generation
                        await self.stop_generation()
                            
                        # Start new generation
                        prompt = data.get("prompt", "Once upon a time")
                        temperature = float(data.get("temperature", 0.7))
                        
                        # Create generation task
                        self.current_generation_task = asyncio.create_task(
                            self.run_generation(prompt, temperature)
                        )
                        self.tasks.append(self.current_generation_task)
                        
                    elif command == "stop":
                        # Stop the generator directly
                        self.generator.stop()
                        # Also cancel the task
                        await self.stop_generation()
                        # Send stop confirmation to client
                        await websocket.send(json.dumps({"event": "stopped"}))
                            
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Client connection closed")
        finally:
            self.clients.remove(websocket)
            logger.debug(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def stop_generation(self) -> None:
        """Stop the current generation task"""
        # Signal the generator to stop
        self.generator.stop()
        
        # Cancel the task
        if self.current_generation_task and not self.current_generation_task.done():
            self.current_generation_task.cancel()
            try:
                await self.current_generation_task
            except asyncio.CancelledError:
                logger.debug("Generation task cancelled")
            except Exception as e:
                logger.error(f"Error cancelling generation: {str(e)}")
            finally:
                self.current_generation_task = None
            
    async def run_generation(self, prompt: str, temperature: float) -> None:
        """Run token generation in a separate task"""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.generator.generate_tokens,
                prompt,
                temperature
            )
        except asyncio.CancelledError:
            logger.debug("Generation cancelled")
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            
    async def start(self, shutdown_event: asyncio.Event) -> None:
        """Start the WebSocket server"""
        self.server = await websockets.serve(self.handle_client, self.host, self.port)
        logger.info(f"WebSocket server started at ws://{self.host}:{self.port}")
        
        try:
            await shutdown_event.wait()
        finally:
            # Clean shutdown
            logger.info("Shutting down WebSocket server")
            
            # Stop the generator
            self.generator.stop()
            
            # Cancel current generation
            await self.stop_generation()
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Close all client connections
            for client in list(self.clients):
                await client.close(1001, "Server shutting down")
            
            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                
            logger.info("WebSocket server shutdown complete") 