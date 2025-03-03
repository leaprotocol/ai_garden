#!/usr/bin/env python
"""
Example: Infinite Token Generator with Console and WebSocket Output

This example demonstrates continuous token generation with both
console output and WebSocket streaming to web clients.
"""

import sys
import asyncio
import logging
import threading
import signal
import uvicorn
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Set up logging with different levels for file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/infinite_tokens.log"),
    ]
)

# Add console handler with INFO level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Global variables for clean shutdown
http_server = None
websocket_server = None
shutdown_event = asyncio.Event()
console = Console()

def start_http_server():
    """Start the HTTP server in a separate thread"""
    global http_server
    from http_server import app
    config = uvicorn.Config(app, host="localhost", port=8000, log_level="error")
    http_server = uvicorn.Server(config)
    http_server.run()

async def shutdown():
    """Clean shutdown sequence"""
    logger.info("Starting shutdown sequence...")
    console.print("\n[yellow]Shutting down...[/yellow]")
    
    # Stop the HTTP server
    if http_server:
        http_server.should_exit = True
        logger.info("HTTP server shutdown initiated")
    
    # Stop the WebSocket server and generator
    if websocket_server:
        try:
            websocket_server.generator.stop()
            await websocket_server.stop_generation()
            logger.info("WebSocket server and generator stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
    
    # Final cleanup
    logger.info("Shutdown complete")
    console.print("[green]Shutdown complete[/green]")

def signal_handler():
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    console.print("\n[yellow]Shutdown signal received (Ctrl+C)[/yellow]")
    
    # Set the shutdown event
    shutdown_event.set()

async def main():
    from token_generator import TokenGenerator
    from websocket_server import TokenWebSocketServer
    from console_handler import ConsoleHandler
    
    global websocket_server
    
    # Create components
    generator = TokenGenerator()
    console_handler = ConsoleHandler()
    websocket_server = TokenWebSocketServer()
    
    # Add console handler to generator callbacks
    generator.add_callback(console_handler.handle_token)
    
    # Display startup info
    console.print(Panel.fit(
        "[yellow]Infinite Token Generator with WebSocket Support[/yellow]\n"
        "Press [bold red]Ctrl+C[/bold red] at any time to stop generation.\n"
        "Web interface available at http://localhost:8000\n"
        "WebSocket server available at ws://localhost:8765"
    ))
    
    # Log device information
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    console.print(f"Using device: {device}")
    
    try:
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=start_http_server)
        http_thread.daemon = True
        http_thread.start()
        
        # Start WebSocket server
        server_task = asyncio.create_task(websocket_server.start(shutdown_event))
        
        # Wait for shutdown event
        await shutdown_event.wait()
        
        # Run shutdown sequence
        await shutdown()
        
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
            
        except Exception as e:
        logger.error(f"Error: {str(e)}")
        console.print(f"\n\n[bold red]Error:[/bold red] {str(e)}")
    finally:
        # Final cleanup and exit
        logger.info("Exiting application")
        sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        console.print("\n[red]Forced exit[/red]")
    finally:
        loop.close() 