import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import logging
import gc
import os

logger = logging.getLogger(__name__)

class StateCacher:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct", device: str = "cuda"):
        logger.debug(f"Loading model '{model_name}' on device '{device}'.")
        self.device = device
        self.model_name = model_name

        os.makedirs("saved_states", exist_ok=True)

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared CUDA cache")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.info(f"Model loaded successfully on {device}")

            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            logger.info(f"Model size: {model_size:.2f}M parameters")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def process_and_cache(self, initial_text: str, save_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        logger.debug(f"Processing initial text: {initial_text}")
        logger.info(f"Processing text and caching state for: {initial_text[:100]}...")

        try:
            inputs = self.tokenizer(
                initial_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            input_length = inputs.input_ids.shape[1]
            logger.info(f"Input length: {input_length} tokens")

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True
                )
                logger.debug(f"past_key_values length: {len(outputs.past_key_values)}")
                logger.debug(f"Shape of past_key_values[0][0]: {outputs.past_key_values[0][0].shape}")

                if save_path:
                    state_dict = {
                        'past_key_values': outputs.past_key_values,
                        'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask
                    }
                    torch.save(state_dict, os.path.join("saved_states", save_path))
                    logger.info(f"Saved state to {save_path}")

                gen_outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
                logger.debug(f"Generated sequences: {gen_outputs.sequences.shape}")

            generated_text = self.tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True)
            logger.info(f"Generated text length: {len(generated_text.split())} words")

            cached_state = {
                'cache': outputs.past_key_values,
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }

            return generated_text, cached_state

        except Exception as e:
            logger.error(f"Error in process_and_cache: {str(e)}", exc_info=True)
            raise

    def generate_continuation(
        self,
        cached_state: Dict[str, Any],
        suffix: str,
        max_new_tokens: int = 50,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate continuation based on a suffix using cached state with optional seeding.
        
        Args:
            cached_state: Dictionary containing 'cache', 'input_ids', and 'attention_mask'
            suffix: The suffix to append and generate continuation from
            max_new_tokens: Maximum number of new tokens to generate
            seed: Optional seed for reproducibility
        
        Returns:
            The generated continuation as a string
        """
        logger.debug(f"Generating continuation with suffix: '{suffix}' and seed: {seed}")
        logger.info(f"Generating continuation for suffix: {suffix[:50]}... with seed: {seed}")

        try:
            # Retrieve original input_ids and attention_mask
            original_input_ids = cached_state['input_ids']
            original_attention_mask = cached_state['attention_mask']
            past_key_values = cached_state['cache']

            logger.debug(f"Original input_ids shape: {original_input_ids.shape}")
            logger.debug(f"Original attention_mask shape: {original_attention_mask.shape}")
            logger.debug(f"Length of past_key_values: {len(past_key_values)}")
            logger.debug(f"Shape of past_key_values[0][0]: {past_key_values[0][0].shape}")

            # Tokenize the suffix
            suffix_inputs = self.tokenizer(
                suffix,
                return_tensors="pt"
            ).to(self.device)

            suffix_ids = suffix_inputs.input_ids
            suffix_attention_mask = suffix_inputs.attention_mask

            logger.debug(f"Suffix input_ids shape: {suffix_ids.shape}")
            logger.debug(f"Suffix attention_mask shape: {suffix_attention_mask.shape}")

            # Concatenate original input_ids with suffix_ids
            new_input_ids = torch.cat([original_input_ids, suffix_ids], dim=1)
            new_attention_mask = torch.cat([original_attention_mask, suffix_attention_mask], dim=1)

            logger.debug(f"New input_ids shape: {new_input_ids.shape}")
            logger.debug(f"New attention_mask shape: {new_attention_mask.shape}")

            # Set up the generator with the specified seed for reproducibility
            generator = None
            if seed is not None:
                logger.debug(f"Setting seed for generation: {seed}")
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Generate continuation
            gen_outputs = self.model.generate(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                generator=generator  # Incorporate the generator with seed
            )
            
            logger.debug(f"Generated sequences shape: {gen_outputs.shape}")

            # Decode the generated sequences
            continuation = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            logger.info(f"Generated continuation length: {len(continuation.split())} words")

            return continuation

        except Exception as e:
            logger.error(f"Error in generate_continuation: {str(e)}", exc_info=True)
            raise
