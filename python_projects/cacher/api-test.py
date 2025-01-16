import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = 'HuggingFaceTB/SmolLM2-360M-Instruct'  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the input text
input_text = "Your input text here"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

# Move tensors to device
device = 'cpu'
input_ids = input_ids.to(device)
model.to(device)

# Perform forward pass
with torch.no_grad():
    outputs = model(input_ids)

# Get the logits and convert to probabilities
logits = outputs.logits
probabilities = torch.softmax(logits[0], dim=-1)

# Get the most likely next tokens
top_k = 5  # Show top 5 predictions
top_probs, top_indices = torch.topk(probabilities[-1], top_k)

print("\nInput text:", input_text)
print("\nMost likely next tokens:")
for prob, idx in zip(top_probs, top_indices):
    token = tokenizer.decode([idx])
    percentage = prob.item() * 100
    print(f"Token: '{token}' - Probability: {percentage:.2f}%")

# Also show the full predicted next sequence
generated = model.generate(
    input_ids,
    max_new_tokens=20,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id
)
full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\nFull generated response:")
print(full_response)
