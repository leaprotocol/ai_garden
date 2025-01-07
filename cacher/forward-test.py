from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import torch.nn.functional as F
import logging
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model (on CPU)
logger.info("Loading model...")
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to('cpu')

# Ensure the cache directory exists
cache_dir = "saved_caches"
os.makedirs(cache_dir, exist_ok=True)

# --- First forward pass & save cache ---
text1 = "The capital of France is"
inputs1 = tokenizer(text1, return_tensors="pt").to('cpu')

with torch.no_grad():
    outputs1 = model(
        **inputs1,
        use_cache=True
    )

cache = DynamicCache.from_legacy_cache(outputs1.past_key_values)
cache_file_path = os.path.join(cache_dir, "cache_after_first_pass.pt")
torch.save(cache, cache_file_path)
logger.info(f"Cache saved to {cache_file_path}")

# --- Load cache for the second pass ---
loaded_cache = torch.load(cache_file_path, map_location='cpu')
past_key_values = loaded_cache.to_legacy_cache()

# Second forward pass, reusing the past state
text2 = " Paris. The city"
inputs2 = tokenizer(text2, return_tensors="pt").to('cpu')  # Define inputs2 here

# --- Adjust attention mask for second pass ---
past_length = past_key_values[0][0].shape[2]
new_attention_mask = torch.ones(
    inputs2.input_ids.shape[0],  # Now inputs2 is defined
    inputs2.input_ids.shape[1] + past_length,
    dtype=torch.long
).to('cpu')

with torch.no_grad():
    outputs2 = model(
        input_ids=inputs2.input_ids,
        attention_mask=new_attention_mask,
        past_key_values=past_key_values,
        use_cache=True
    )

# Get top next tokens after second text
next_token_logits = outputs2.logits[0, -1, :]
probabilities = F.softmax(next_token_logits, dim=-1)
top_probs, top_indices = torch.topk(probabilities, k=5)

print("\nTop next tokens after second text (with reused state):")
for prob, idx in zip(top_probs, top_indices):
    token = tokenizer.decode([idx])
    print(f"Token: '{token}', Prob: {prob.item():.4f}, ID: {idx.item()}")