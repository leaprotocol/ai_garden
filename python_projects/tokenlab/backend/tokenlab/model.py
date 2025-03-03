import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, AsyncIterator
import numpy as np

class TokenAnalyzer:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        print(f"Loading tokenizer for: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        print(f"Model loaded on device: {self.device}")

    async def generate_stream(self, prompt: str, max_length: int = 5) -> AsyncIterator[Dict]:
        """Generate text and stream token probabilities."""
        try:
            # First, tokenize the entire prompt and get probabilities
            initial_tokens = await self.get_token_probabilities(prompt)
            
            # Yield initial token structure
            yield {
                'type': 'initial_tokens',
                'tokens': initial_tokens
            }
            
            # Tokenize input and get initial context for generation
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            
            # Generate additional tokens
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Get top 5 predictions
                    top_probs, top_indices = torch.topk(probs, k=5)
                    next_token_id = top_indices[0]
                    
                    token_data = {
                        'type': 'next_token',
                        'token': self.tokenizer.decode([next_token_id]),
                        'probability': float(top_probs[0]),
                        'top_predictions': [
                            {'token': self.tokenizer.decode([idx]), 
                             'probability': float(prob)}
                            for idx, prob in zip(top_indices, top_probs)
                        ]
                    }
                    
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    yield token_data
                    
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
                        
        except Exception as e:
            print(f"Error generating text: {e}")
            return

    async def get_token_probabilities(self, text: str) -> List[Dict]:
        """Get probability distribution for each token in the text."""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            tokens = [self.tokenizer.decode([id]) for id in inputs.input_ids[0]]

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
                probs = torch.softmax(logits, dim=-1)

            # Process each token position
            token_data = []
            for pos, (token_id, token_text) in enumerate(zip(inputs.input_ids[0], tokens)):
                # Get top 5 predictions for this position
                top_probs, top_indices = torch.topk(probs[pos], k=5)
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
                
                # Calculate actual token probability
                token_prob = float(probs[pos, token_id].item())
                
                token_data.append({
                    'position': pos,
                    'token': token_text,
                    'probability': token_prob,
                    'top_predictions': [
                        {'token': token, 'probability': float(prob)}
                        for token, prob in zip(top_tokens, top_probs)
                    ]
                })

            return token_data

        except Exception as e:
            print(f"Error analyzing tokens: {e}")
            return [] 