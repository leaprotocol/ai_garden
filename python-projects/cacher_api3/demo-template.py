import asyncio
import logging
from rich.console import Console
from rich.logging import RichHandler
from typing import Optional, Dict, Any

# Initialize Rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("demo")

class CacherDemo:
    """Template class for creating Cacher API demos."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the demo with configuration."""
        from cacher_api3.demo4 import CacherClientV4
        self.client = CacherClientV4(base_url)
        self.states = {}  # Track created states
        
    async def setup(self):
        """Setup demo resources and configurations."""
        logger.info("Setting up demo resources...")
        # Add any setup code here
        
    async def cleanup(self):
        """Cleanup resources after demo completion."""
        logger.info("Cleaning up resources...")
        # Delete any created states
        for state_id in self.states:
            try:
                await self.client.delete_state(state_id)
            except Exception as e:
                logger.warning(f"Failed to delete state {state_id}: {e}")
        await self.client.close()
        
    async def create_demo_state(
        self,
        text: str,
        model_id: str = "gpt2",
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create and track a new state."""
        state = await self.client.create_state(model_id, text, config)
        state_id = state["state_id"]
        self.states[state_id] = state
        return state_id
        
    async def run_basic_generation(self):
        """Demonstrate basic text generation."""
        logger.info("[bold blue]Basic Text Generation Demo[/bold blue]")
        
        # Create state with initial text
        state_id = await self.create_demo_state(
            "The quick brown fox",
            config={"temperature": 0.7, "top_p": 0.9}
        )
        
        # Generate continuation
        result = await self.client.generate_token(
            state_id,
            max_tokens=10,
            temperature=0.7
        )
        
        logger.info(f"Generated text: {result['text']}")
        
    async def run_reproducible_generation(self):
        """Demonstrate reproducible text generation."""
        logger.info("[bold blue]Reproducible Generation Demo[/bold blue]")
        
        # Create two states with same seed
        seed = 42
        text = "Once upon a time"
        config = {"temperature": 0.7, "top_p": 0.9, "seed": seed}
        
        state_id1 = await self.create_demo_state(text, config=config)
        state_id2 = await self.create_demo_state(text, config=config)
        
        # Generate with same parameters
        result1 = await self.client.generate_token(
            state_id1,
            max_tokens=5,
            temperature=0.7,
            seed=seed
        )
        
        result2 = await self.client.generate_token(
            state_id2,
            max_tokens=5,
            temperature=0.7,
            seed=seed
        )
        
        # Verify outputs match
        if result1["text"] == result2["text"]:
            logger.info("[bold green]✓ Outputs match![/bold green]")
        else:
            logger.warning("[bold red]✗ Outputs differ![/bold red]")
            
    async def run_token_analysis(self):
        """Demonstrate token probability analysis."""
        logger.info("[bold blue]Token Analysis Demo[/bold blue]")
        
        state_id = await self.create_demo_state("The quick brown")
        
        # Get top token predictions
        top_tokens = await self.client.get_top_n_tokens(state_id, n=5)
        
        # Display results in table
        from rich.table import Table
        table = Table(title="Top 5 Predicted Tokens")
        table.add_column("Token", style="cyan")
        table.add_column("Probability", style="magenta")
        
        for token in top_tokens.get("tokens", []):
            table.add_row(
                token["token"],
                f"{token['probability']:.4f}"
            )
            
        console.print(table)

async def main():
    """Main demo runner."""
    demo = CacherDemo()
    
    try:
        await demo.setup()
        
        # Run demo sections
        await demo.run_basic_generation()
        await demo.run_reproducible_generation()
        await demo.run_token_analysis()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 