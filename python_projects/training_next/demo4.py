import logging
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
console = Console()

def analyze_token_presence(text: str, target: str, tokenizer) -> dict:
    """Analyze if target token appears in the input context."""
    # Split into context and target
    words = text.split()
    context = ' '.join(words[:-1])
    
    # Get token IDs for target and context
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    
    # Check if target token appears in context
    target_in_context = False
    for i in range(len(context_ids) - len(target_ids) + 1):
        if context_ids[i:i+len(target_ids)] == target_ids:
            target_in_context = True
            break
    
    # Get individual tokens for more detailed analysis
    context_tokens = tokenizer.tokenize(context)
    target_tokens = tokenizer.tokenize(target)
    
    return {
        'target': target,
        'target_ids': target_ids,
        'target_tokens': target_tokens,
        'appears_in_context': target_in_context,
        'context_length': len(context_tokens),
        'target_length': len(target_tokens)
    }

def main():
    try:
        # Initialize
        logger.info("Starting LAMBADA token presence analysis...")
        MODEL_NAME = "prajjwal1/bert-small"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load dataset
        SAMPLE_SIZE = 100  # Analyze first 100 examples
        logger.info(f"Loading LAMBADA dataset (first {SAMPLE_SIZE} examples)")
        dataset = load_dataset("lambada", split=f"validation[:{SAMPLE_SIZE}]")
        
        # Analysis metrics
        metrics = {
            'total': 0,
            'appears_in_context': 0,
            'single_token_targets': 0,
            'multi_token_targets': 0,
            'max_target_tokens': 0,
            'total_context_tokens': 0
        }
        
        # Examples where target appears in context
        examples_with_target = []
        
        # Process each example
        for i, example in enumerate(dataset, 1):
            text = example['text']
            words = text.split()
            target = words[-1]
            
            results = analyze_token_presence(text, target, tokenizer)
            metrics['total'] += 1
            
            if results['appears_in_context']:
                metrics['appears_in_context'] += 1
                examples_with_target.append({
                    'index': i,
                    'target': target,
                    'text': text
                })
            
            if len(results['target_tokens']) == 1:
                metrics['single_token_targets'] += 1
            else:
                metrics['multi_token_targets'] += 1
            
            metrics['max_target_tokens'] = max(metrics['max_target_tokens'], 
                                             len(results['target_tokens']))
            metrics['total_context_tokens'] += results['context_length']
            
            # Progress update every 10 examples
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(dataset)} examples...")
        
        # Display results
        console.print("\n[bold green]Analysis Results[/bold green]")
        
        # Statistics table
        stats_table = Table(show_lines=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        stats_table.add_row("Total Examples", str(metrics['total']))
        stats_table.add_row("Target Appears in Context", str(metrics['appears_in_context']))
        stats_table.add_row("Single-Token Targets", str(metrics['single_token_targets']))
        stats_table.add_row("Multi-Token Targets", str(metrics['multi_token_targets']))
        stats_table.add_row("Max Target Tokens", str(metrics['max_target_tokens']))
        stats_table.add_row("Avg Context Length", 
                          f"{metrics['total_context_tokens']/metrics['total']:.1f}")
        stats_table.add_row("Target in Context %", 
                          f"{100*metrics['appears_in_context']/metrics['total']:.1f}%")
        
        console.print(stats_table)
        
        # Display some examples where target appears in context
        if examples_with_target:
            console.print("\n[bold yellow]Examples where target appears in context:[/bold yellow]")
            examples_table = Table(show_lines=True)
            examples_table.add_column("Index", style="cyan", width=6)
            examples_table.add_column("Target", style="magenta", width=15)
            examples_table.add_column("Text", style="green")
            
            for example in examples_with_target[:5]:  # Show first 5 examples
                examples_table.add_row(
                    str(example['index']),
                    example['target'],
                    example['text'][:100] + "..."  # Truncate long texts
                )
            
            console.print(examples_table)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        logger.info("Analysis completed")

if __name__ == "__main__":
    main() 