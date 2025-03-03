"""
Console handler for displaying tokens in the terminal.
"""

import sys
import logging
from rich.console import Console
from rich.text import Text
from typing import Any

logger = logging.getLogger(__name__)

class ConsoleHandler:
    def __init__(self):
        self.console = Console()
        self.token_count = 0
        self.stats_interval = 100  # Show stats every 100 tokens instead of 20
        
    def handle_token(self, event_data: dict[str, Any]) -> None:
        """Handle a token event by displaying it in the console"""
        try:
            # Check for special events
            if "event" in event_data:
                event_type = event_data["event"]
                
                if event_type == "stopped":
                    self.handle_stop(event_data.get("token_count", self.token_count))
                    return
                    
                if event_type == "error":
                    self.handle_error(event_data.get("error", "Unknown error"))
                    return
            
            # Extract data
            token = event_data["token"]
            prob_color = event_data["probability_color"]
            token_count = event_data["token_count"]
            
            # Update token count
            self.token_count = token_count
            
            # Create and style token text
            token_text = Text(token)
            token_text.stylize(f"bold {prob_color}")
            
            # Print the token
            self.console.print(token_text, end="", highlight=False)
            sys.stdout.flush()
            
            # Display token count and top probability less frequently
            if token_count % self.stats_interval == 0:
                top_token = event_data["top_tokens"][0]  # Just show the top token
                self.console.print(f"\n[dim][#{token_count} | Top: '{top_token['token']}' ({top_token['probability']:.2f})][/dim]")
                
        except Exception as e:
            logger.error(f"Error in console handler: {str(e)}")
            
    def handle_error(self, error: str) -> None:
        """Display an error message"""
        self.console.print(f"\n\n[bold red]Error:[/bold red] {error}")
        
    def handle_stop(self, token_count: int) -> None:
        """Display stop message with token count"""
        self.console.print("\n\n[bold yellow]Generation stopped.[/bold yellow]")
        self.console.print(f"[bold blue]Total tokens:[/bold blue] {token_count}")
        logger.info(f"Generator stopped after {token_count} tokens")