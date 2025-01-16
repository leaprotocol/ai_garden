from transformers import pipeline
import torch
from tabulate import tabulate

# Check if CUDA is available and supported
device = "cuda" if torch.cuda.is_available() and \
    torch.cuda.get_device_capability()[0] >= 3 and \
    torch.cuda.get_device_capability()[1] >= 7 else "cpu"

print(f"Using device: {device}")

classifier = pipeline("fill-mask", device=device)
input_text = "Paris is the capital of <mask>."
x = classifier(input_text)

print(f"\nInput text: {input_text}")

# Handle both single and multiple mask cases
predictions_list = [x] if not isinstance(x[0], list) else x

# Create tables for each mask
for mask_index, predictions in enumerate(predictions_list):
    # Prepare table data
    table_data = [
        [i+1, pred['token_str'].strip(), f"{pred['score']*100:.1f}%"]
        for i, pred in enumerate(predictions)
    ]
    
    print(f"\nPredictions for mask #{mask_index + 1}:")
    print(tabulate(
        table_data,
        headers=['Rank', 'Token', 'Confidence'],
        tablefmt='grid'
    ))

# Print complete sentences in table format
print("\nPossible complete sentences:")
combinations = []

if len(predictions_list) == 1:
    # Single mask case
    for pred in predictions_list[0][:3]:  # Show top 3 possibilities
        filled_text = input_text.replace("<mask>", pred['token_str'].strip())
        combinations.append([filled_text, f"{pred['score']*100:.2f}%"])
else:
    # Multiple mask case
    for first in predictions_list[0][:2]:
        for second in predictions_list[1][:2]:
            filled_text = input_text.replace("<mask>", first['token_str'].strip(), 1)
            filled_text = filled_text.replace("<mask>", second['token_str'].strip(), 1)
            confidence = (first['score'] * second['score']) * 100
            combinations.append([filled_text, f"{confidence:.2f}%"])

print(tabulate(
    combinations,
    headers=['Sentence', 'Confidence'],
    tablefmt='grid'
))