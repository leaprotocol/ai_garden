import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from cacher_api2.models import ModelManager
from cacher_api2.services.cache_service import CacheService
from cacher_api2.utils import get_logger, generate_cache_id
from typing import Dict, Tuple, Optional, List

logger = get_logger(__name__)

def _convert_cache_to_dict(past_key_values):
    logger.debug(f"Bypassing cache conversion. Input type: {type(past_key_values)}")
    return past_key_values

class GenerationService:
    def __init__(self, model_manager: ModelManager, cache_service: CacheService):
        self.model_manager = model_manager
        self.cache_service = cache_service

    async def forward(self, model_id: str, text: str, cache_id: Optional[str] = None) -> Tuple[str, int]:
        logger.info(f"Forward pass for model {model_id} with text: {text[:50]}...")
        model: PreTrainedModel = self.model_manager.get_model(model_id)
        tokenizer: PreTrainedTokenizer = self.model_manager.get_tokenizer(model_id)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        input_length = inputs.input_ids.shape[1]

        if cache_id:
            # Load existing cache
            if cache_id not in self.cache_service.cached_states:
                raise ValueError("Cache not found")

            logger.info(f"Using the cached state {cache_id}")
            cached_state = self.cache_service.cached_states[cache_id]

            # Perform a forward pass to update the cache
            with torch.no_grad():
                logger.debug(f"Past key values type before forward: {type(cached_state['past_key_values'])}")
                outputs = model(
                    **inputs,
                    past_key_values=cached_state['past_key_values'],
                    use_cache=True
                )
                logger.debug(f"Past key values type after forward: {type(outputs.past_key_values)}")

            # Update the cache
            logger.debug(f"Past key values type before (no) conversion: {type(outputs.past_key_values)}")
            converted_past_key_values = _convert_cache_to_dict(outputs.past_key_values)
            logger.debug(f"Past key values type after (no) conversion: {type(converted_past_key_values)}")
            updated_cache = {
                'past_key_values': converted_past_key_values,
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }
            self.cache_service.add_to_cache(cache_id, updated_cache)
        else:
            # Create a new cache
            logger.info(f"Creating a new cache")
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True
                )

            logger.debug(f"Past key values type before (no) conversion (new cache): {type(outputs.past_key_values)}")
            converted_past_key_values = _convert_cache_to_dict(outputs.past_key_values)
            logger.debug(f"Past key values type after (no) conversion (new cache): {type(converted_past_key_values)}")
            new_cache = {
                'past_key_values': converted_past_key_values,
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }
            cache_id = generate_cache_id()
            self.cache_service.add_to_cache(cache_id, new_cache)

        return cache_id, input_length

    async def get_next_token_probs(self, model_id: str, cache_id: str) -> Dict[str, float]:
        logger.info(f"Getting next token probabilities for model {model_id} with cache {cache_id}")
        model: PreTrainedModel = self.model_manager.get_model(model_id)
        tokenizer: PreTrainedTokenizer = self.model_manager.get_tokenizer(model_id)

        if cache_id not in self.cache_service.cached_states:
            raise ValueError("Cache not found")

        cache = self.cache_service.cached_states[cache_id]

        with torch.no_grad():
            outputs = model(
                input_ids=cache['input_ids'],
                attention_mask=cache['attention_mask'],
                past_key_values=cache['past_key_values'],
                use_cache=True
            )

        next_token_logits = outputs.logits[:, -1, :]
        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k=5)

        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()

        probs_dict = {
            tokenizer.decode(idx): prob
            for idx, prob in zip(top_indices, top_probs)
        }

        return probs_dict

    async def beam_search(self, model_id: str, cache_id: str, beam_width: int, max_length: int) -> List[str]:
        logger.info(f"Performing beam search for model {model_id} with cache {cache_id}")
        model: PreTrainedModel = self.model_manager.get_model(model_id)
        tokenizer: PreTrainedTokenizer = self.model_manager.get_tokenizer(model_id)

        if cache_id not in self.cache_service.cached_states:
            raise ValueError("Cache not found")

        cache = self.cache_service.cached_states[cache_id]
        
        # Prepare inputs for beam search
        input_ids = cache['input_ids']
        attention_mask = cache['attention_mask']
        past_key_values = cache['past_key_values']

        # Perform beam search
        beam_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_length=max_length,
            num_beams=beam_width,
            early_stopping=True,
            num_return_sequences=beam_width,  # Return top sequences
            no_repeat_ngram_size=2,  # Optional: Prevent repeating n-grams
            use_cache=True
        )

        # Decode generated sequences
        generated_sequences = [
            tokenizer.decode(seq, skip_special_tokens=True)
            for seq in beam_output
        ]

        return generated_sequences

    async def get_greedy_token(self, model_id: str, cache_id: str) -> Tuple[str, float]:
        logger.info(f"Getting greedy token for model {model_id} with cache {cache_id}")
        model: PreTrainedModel = self.model_manager.get_model(model_id)
        tokenizer: PreTrainedTokenizer = self.model_manager.get_tokenizer(model_id)

        if cache_id not in self.cache_service.cached_states:
            raise ValueError("Cache not found")

        cache = self.cache_service.cached_states[cache_id]

        with torch.no_grad():
            outputs = model(
                input_ids=cache['input_ids'],
                attention_mask=cache['attention_mask'],
                past_key_values=cache['past_key_values'],
                use_cache=True
            )

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        probability = torch.softmax(next_token_logits, dim=-1).squeeze()[next_token].item()

        return tokenizer.decode(next_token), probability 