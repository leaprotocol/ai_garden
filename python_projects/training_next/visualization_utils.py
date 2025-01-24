"""Utilities for visualization of analysis results."""

from typing import Callable, Dict, List, Any, Tuple
import torch
from rich.text import Text

# First, let's define the HTML template with proper structure
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: 'Courier New', monospace;
            margin: 20px;
            background: #f8f9fa;
        }}
        .sentence {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .sentence h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .sentence p {{
            margin: 0;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        td, th {{ 
            padding: 8px 12px; 
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        th {{ 
            background: #f1f3f5;
            font-weight: bold;
            border-bottom: 2px solid #ddd;
        }}
        .token {{ 
            font-weight: bold; 
            text-align: left;
            font-family: monospace;
        }}
        .context {{ 
            color: #666; 
            text-align: left;
            font-family: monospace;
            white-space: pre-wrap;
        }}
        tr:hover {{ 
            background-color: #f5f5f5; 
        }}
        .prob {{ min-width: 60px; }}
        .entropy {{ min-width: 70px; }}
        .loss {{ min-width: 70px; }}
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

HTML_ROW = """
            <tr>
                {cells}
            </tr>"""

def create_header(columns: List[str]) -> str:
    """Create HTML table header from column names."""
    return HTML_TEMPLATE.format(
        headers="\n                ".join(f"<th>{col}</th>" for col in columns)
    )

def create_row(cells: List[Tuple[str, str]]) -> str:
    """Create HTML table row from (value, style) pairs."""
    formatted_cells = [
        f'<td style="{style}">{value}</td>'
        for value, style in cells
    ]
    return HTML_ROW.format(cells="\n                ".join(formatted_cells))

def format_number(value: float, precision: int = 2) -> str:
    """Format a number with specified precision."""
    return f"{value:.{precision}f}"

def create_color_formatter(
    get_color: Callable[[float], str],
    precision: int = 2
) -> Callable[[float], Tuple[str, str]]:
    """Create a formatter function that returns (formatted_value, style)."""
    def formatter(value: float) -> Tuple[str, str]:
        return (format_number(value, precision), f"color: {get_color(value)}")
    return formatter

def create_token_formatter(
    get_color: Callable[[float, float], str]
) -> Callable[[str, float, float], Tuple[str, str]]:
    """Create a formatter for token cells that includes color based on probabilities."""
    def formatter(token: str, forward_prob: float, backward_prob: float) -> Tuple[str, str]:
        return (token, f"color: {get_color(forward_prob, backward_prob)}")
    return formatter

def create_context_formatter() -> Callable[[str], Tuple[str, str]]:
    """Create a formatter for context cells."""
    def formatter(context: str) -> Tuple[str, str]:
        return (context, "color: #666; text-align: left")
    return formatter

def create_html_table(results, bert_tokenizer, smol_tokenizer):
    """Create a properly structured HTML table."""
    table = '<table>\n<thead>\n<tr>'
    
    # Headers
    for col in results['columns']:
        css_class = col.lower().replace(' ', '_')
        table += f'<th class="{css_class}">{col}</th>'
    table += '</tr>\n</thead>\n<tbody>'
    
    # Rows
    for example in results['examples']:
        bert_tokens = example['bert_tokens']
        smol_tokens = example['smol_tokens']
        
        # Align tokens by sentence position
        aligned_tokens = align_tokens_by_position(bert_tokens, smol_tokens)
        
        for token_pair in aligned_tokens:
            bert_token, smol_token = token_pair
            
            # Get BERT probabilities and characteristics
            bert_prob = bert_token['prob']
            bert_neighbor_prob = bert_token['neighbor_prob']
            bert_entropy = bert_token['entropy']
            bert_loss = bert_token['loss']
            
            # Get SmolLM probabilities and characteristics
            smol_prob = smol_token['prob']
            smol_entropy = smol_token['entropy']
            smol_loss = smol_token['loss']
            
            # Create row with both tokenizations and their characteristics
            row = [
                # BERT token info
                bert_token['token_text'],
                f"{bert_prob:.2f}",
                f"{bert_neighbor_prob:.2f}",
                f"{bert_entropy:.2f}",
                f"{bert_loss:.2f}",
                
                # SmolLM token info
                smol_token['token_text'],
                f"{smol_prob:.2f}",
                f"{smol_entropy:.2f}",
                f"{smol_loss:.2f}",
                
                # Context
                example['context']
            ]
            
            # Add row to table
            table += '\n<tr>'
            for col in results['columns']:
                css_class = col.lower().replace(' ', '_')
                if col in results['formatters']:
                    value, style = results['formatters'][col](row[results['columns'].index(col)])
                    # Remove any rich text formatting
                    if isinstance(value, str):
                        value = value.replace('[bold red]', '').replace('[/bold red]', '')
                    table += f'<td class="{css_class}" style="{style}">{value}</td>'
                else:
                    table += f'<td class="{css_class}">{row[results['columns'].index(col)]}</td>'
            table += '</tr>'
    
    table += '\n</tbody>\n</table>'
    
    # Wrap in complete HTML document
    return HTML_TEMPLATE.format(content=table)

def align_tokens_by_position(bert_tokens, smol_tokens):
    """
    Align tokens from different tokenizers by their position in the sentence.
    Returns list of (bert_token, smol_token) pairs.
    """
    aligned = []
    bert_idx = 0
    smol_idx = 0
    
    while bert_idx < len(bert_tokens) and smol_idx < len(smol_tokens):
        bert_token = bert_tokens[bert_idx]
        smol_token = smol_tokens[smol_idx]
        
        # Get token positions
        bert_start = bert_token['start']
        bert_end = bert_token['end']
        smol_start = smol_token['start']
        smol_end = smol_token['end']
        
        # If tokens overlap, pair them
        if bert_start <= smol_end and smol_start <= bert_end:
            aligned.append((bert_token, smol_token))
            bert_idx += 1
            smol_idx += 1
        elif bert_start < smol_start:
            # BERT token is before SmolLM token
            aligned.append((bert_token, None))
            bert_idx += 1
        else:
            # SmolLM token is before BERT token
            aligned.append((None, smol_token))
            smol_idx += 1
    
    return aligned

# Color utilities
def get_color_for_value(value: float, channel: str = 'r', min_val: float = 0.0, max_val: float = 1.0) -> str:
    """Convert a value to a color intensity in the specified RGB channel."""
    # Clamp value between min and max
    normalized = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    # Create RGB color with the specified channel
    r = int(255 * normalized) if channel == 'r' else 0
    g = int(255 * normalized) if channel == 'g' else 0
    b = int(255 * normalized) if channel == 'b' else 0
    
    return f"rgb({r}, {g}, {b})"

def get_prob_color(prob: float, is_backward: bool = False) -> str:
    """Get color for probability value - green for forward, red for backward."""
    channel = 'r' if is_backward else 'g'
    return get_color_for_value(prob, channel)

def get_entropy_color(entropy: float, max_entropy: float = 10.0) -> str:
    """Get blue-scale color for entropy value."""
    return get_color_for_value(entropy, 'b', max_val=max_entropy)

def get_loss_color(loss: float, max_loss: float = 15.0) -> str:
    """Get blue-scale color for loss value."""
    return get_color_for_value(loss, 'b', max_val=max_loss)

def get_ratio_color(ratio: float, is_backward: bool = False) -> str:
    """Get color for ratio value - green for forward, red for backward."""
    # Normalize ratio to 0-1 range, assuming ratio is between 0.1 and 10.0
    normalized = (min(max(ratio, 0.1), 10.0) - 0.1) / 9.9
    
    if is_backward:
        # Red channel for backward
        return f"rgb({int(255 * normalized)}, 0, 0)"
    else:
        # Green channel for forward
        return f"rgb(0, {int(255 * normalized)}, 0)"

def get_color_style(f_prob: float, b_prob: float) -> str:
    """Get combined color style based on forward and backward probabilities."""
    # Use red and green channels to represent backward and forward probs
    r = int(255 * b_prob)  # Red for backward
    g = int(255 * f_prob)  # Green for forward
    return f"rgb({r}, {g}, 0)"

def create_console_formatter(
    get_color: Callable[[float], str],
    precision: int = 2,
    width: int = 8
) -> Callable[[float], Tuple[str, str]]:
    """Create a formatter function for console output that returns (formatted_value, style)."""
    def formatter(value: float) -> Tuple[str, str]:
        return (f"{value:{width}.{precision}f}", get_color(value))
    return formatter

def create_console_token_formatter(
    get_color: Callable[[float, float], str],
    width: int = 15
) -> Callable[[str, float, float], Tuple[str, str]]:
    """Create a formatter for token cells in console output."""
    def formatter(token: str, forward_prob: float, backward_prob: float) -> Tuple[str, str]:
        return (f"{token:<{width}}", get_color(forward_prob, backward_prob))
    return formatter

def create_console_context_formatter(
    width: int = 30
) -> Callable[[str], Tuple[str, str]]:
    """Create a formatter for context cells in console output."""
    def formatter(context: str) -> Tuple[str, str]:
        return (f"{context:<{width}}", "dim")
    return formatter

def create_console_row(cells: List[Tuple[str, str]], console) -> Text:
    """Create a rich Text object for console output from (value, style) pairs."""
    text = Text()
    for value, style in cells:
        text.append(value, style=style)
    return text

def create_console_table(
    rows_data: List[Dict[str, Any]],
    columns: List[str],
    formatters: Dict[str, Callable],
    console
) -> None:
    """Create a console table using the provided data, columns and formatters."""
    # Default column widths
    column_widths = {
        'f_prob': 8, 'f_neigh': 8, 'f_ratio': 6, 'f_ent': 6, 'f_loss': 6,
        'token': 15,
        'b_prob': 8, 'b_neigh': 8, 'b_ratio': 6, 'b_ent': 6, 'b_loss': 6,
        's_prob': 8, 's_neigh': 8, 's_ratio': 6, 's_ent': 6, 's_loss': 6,
        'token_id': 6,
        'context': 30
    }
    
    # Print header
    header_text = Text()
    for col in columns:
        width = column_widths.get(col, 8)  # Default to 8 if not specified
        header_text.append(f"{col:>{width}} ")
    console.print(header_text)
    
    # Print rows
    for row in rows_data:
        cells = []
        for col in columns:
            width = column_widths.get(col, 8)  # Default to 8 if not specified
            if col in formatters:
                if col == 'token':
                    # Handle token specially since it needs multiple values
                    token, f_prob, b_prob = row[col]
                    value, style = formatters[col](token, f_prob, b_prob)
                    value = f"{value:<{width}}"
                else:
                    value, style = formatters[col](row[col])
                    if col == 'context':
                        value = f"{value:<{width}}"
                    else:
                        value = f"{value:>{width}}"
            else:
                value = f"{str(row[col]):>{width}}"
                style = ""
            cells.append((value + " ", style))
        console.print(create_console_row(cells, console)) 