# state_cacher.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import logging
import gc
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StateCacher:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct", device: str = "cuda"):
        """Initialize the StateCacher with a specified model.
        
        Args:
            model_name: Name/path of the model to load
            device: Device to run the model on ('cuda' or 'cpu')
        """
        logger.info(f"Initializing StateCacher with model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # Create states directory if it doesn't exist
        os.makedirs("saved_states", exist_ok=True)
        
        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared CUDA cache")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            
            # Load model with appropriate settings for SmolLM2
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.info(f"Model loaded successfully on {device}")
            
            # Log model size
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            logger.info(f"Model size: {model_size:.2f}M parameters")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def process_and_cache(self, initial_text: str, save_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process initial text and cache the model's state.
        
        Args:
            initial_text: The input text to process
            save_path: Optional path to save the state to disk
            
        Returns:
            Tuple of (generated text, cached state)
        """
        logger.info(f"Processing text and caching state for: {initial_text[:100]}...")
        
        try:
            # Tokenize with attention mask
            inputs = self.tokenizer(
                initial_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Prevent too long sequences
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            logger.info(f"Input length: {input_length} tokens")
            
            # First run a forward pass to get the initial state
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True  # Explicitly enable caching
                )
                
                # Save state if path provided
                if save_path:
                    state_dict = {
                        'past_key_values': outputs.past_key_values,
                        'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask
                    }
                    torch.save(state_dict, os.path.join("saved_states", save_path))
                    logger.info(f"Saved state to {save_path}")
                
                # Generate short continuation
                gen_outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,  # Increased from 1 to 50
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            generated_text = self.tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True)
            logger.info(f"Generated text length: {len(generated_text.split())} words")
            
            return generated_text, {
                'past_key_values': outputs.past_key_values,
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }
        
        except Exception as e:
            logger.error(f"Error in process_and_cache: {str(e)}")
            raise
    
    def generate_continuation(
        self,
        cached_state: Dict[str, Any],
        suffix: str,
        max_new_tokens: int = 50
    ) -> str:
        logger.info(f"Generating continuation for suffix: {suffix[:50]}...")

        try:
            # Retrieve original input_ids and attention_mask
            original_input_ids = cached_state['input_ids']      # Shape: [1, 16]
            original_attention_mask = cached_state['attention_mask']  # Shape: [1, 16]

            # Tokenize the suffix
            suffix_inputs = self.tokenizer(
                suffix,
                return_tensors="pt"
            ).to(self.device)

            suffix_ids = suffix_inputs.input_ids                # Shape: [1, 9]
            suffix_attention_mask = suffix_inputs.attention_mask  # Shape: [1, 9]

            logger.info(f"Suffix input_ids shape: {suffix_ids.shape}")
            logger.info(f"Suffix attention_mask shape: {suffix_attention_mask.shape}")

            # Concatenate original input_ids with suffix_ids
            new_input_ids = torch.cat([original_input_ids, suffix_ids], dim=1)  # Shape: [1, 25]
            new_attention_mask = torch.cat([original_attention_mask, suffix_attention_mask], dim=1)  # Shape: [1, 25]

            logger.info(f"New input_ids shape after concatenation: {new_input_ids.shape}")
            logger.info(f"New attention_mask shape after concatenation: {new_attention_mask.shape}")

            # Generate continuation using the updated input_ids and attention_mask
            gen_outputs = self.model.generate(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                max_new_tokens=max_new_tokens,
                past_key_values=cached_state['past_key_values'],
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True  # Ensure structured output
            )

            # Decode the generated sequences
            continuation = self.tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True)
            logger.info(f"Generated continuation length: {len(continuation.split())} words")

            return continuation

        except Exception as e:
            logger.error(f"Error in generate_continuation: {str(e)}")
            logger.error(f"Error details:", exc_info=True)
            raise

    
    def continue_from_cache(
        self,
        cached_state: Dict[str, Any],
        max_new_tokens: int = 50,
        suffix: Optional[str] = None
    ) -> str:
        """
        Continue generation from the cached state, optionally adding a suffix.
        """
        logger.info("Continuing generation from cached state with minimal suffix...")

        try:
            # If no suffix is provided, use a dummy token (EOS token)
            if suffix is None:
                suffix = self.tokenizer.eos_token
            
            # Tokenize the suffix
            suffix_inputs = self.tokenizer(
                suffix,
                return_tensors="pt"
            ).to(self.device)

            suffix_ids = suffix_inputs.input_ids
            suffix_attention_mask = suffix_inputs.attention_mask

            logger.info(f"Suffix input_ids shape: {suffix_ids.shape}")
            logger.info(f"Suffix attention_mask shape: {suffix_attention_mask.shape}")

            # Concatenate the suffix with cached input IDs and attention mask
            new_input_ids = torch.cat([cached_state['input_ids'], suffix_ids], dim=1)
            new_attention_mask = torch.cat([cached_state['attention_mask'], suffix_attention_mask], dim=1)

            logger.info(f"New input_ids shape after concatenation: {new_input_ids.shape}")
            logger.info(f"New attention_mask shape after concatenation: {new_attention_mask.shape}")

            # Generate the continuation
            gen_outputs = self.model.generate(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                past_key_values=cached_state["past_key_values"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True  # Fix: Access structured output
            )

            # Decode the generated sequences
            continuation = self.tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True)
            logger.info(f"Generated continuation length: {len(continuation.split())} words")
            return continuation

        except Exception as e:
            logger.error(f"Error in continue_from_cache: {str(e)}", exc_info=True)
            raise

    
    def generate_large_continuation(self, cached_state: Dict[str, Any], total_new_tokens: int = 1000, batch_size: int = 50) -> str:
        """Generate a large number of tokens efficiently using cached past_key_values.
        
        Args:
            cached_state: Previously cached model state
            total_new_tokens: Total number of new tokens to generate
            batch_size: Number of tokens to generate per batch
            
        Returns:
            Generated continuation as a string
        """
        logger.info(f"Generating a large continuation of {total_new_tokens} tokens in batches of {batch_size}...")
        generated_text = []
        past_key_values = cached_state['past_key_values']
    
        # Start with a minimal suffix to prompt generation
        suffix = " Continue."
    
        for i in range(0, total_new_tokens, batch_size):
            logger.info(f"Generating batch {i // batch_size + 1}...")
    
            # Tokenize the suffix
            suffix_inputs = self.tokenizer(
                suffix,
                return_tensors="pt"
            ).to(self.device)
    
            suffix_ids = suffix_inputs.input_ids                # Shape: [1, X]
            suffix_attention_mask = suffix_inputs.attention_mask  # Shape: [1, X]
    
            # Generate continuation using the cached past_key_values
            gen_outputs = self.model.generate(
                input_ids=suffix_ids,
                attention_mask=suffix_attention_mask,
                max_new_tokens=batch_size,
                past_key_values=past_key_values,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
    
            # Extract the newly generated tokens
            new_tokens = gen_outputs.sequences[:, suffix_ids.shape[-1]:]  # Skip the suffix tokens
            text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            generated_text.append(text)
    
            # Update past_key_values for the next batch
            past_key_values = gen_outputs.past_key_values
    
        return " ".join(generated_text)
    
    def batch_generate_continuations(
        self,
        cached_state: Dict[str, Any],
        suffixes: List[str],
        max_new_tokens: int = 50
    ) -> List[str]:
        """Generate multiple continuations for different suffixes using the same cached state.
        
        Args:
            cached_state: Previously cached model state
            suffixes: List of suffixes to generate continuations for
            max_new_tokens: Maximum number of new tokens per continuation
            
        Returns:
            List of generated continuations
        """
        logger.info(f"Batch generating continuations for {len(suffixes)} suffixes")
        continuations = []
        
        try:
            for i, suffix in enumerate(suffixes, 1):
                logger.info(f"Processing suffix {i}/{len(suffixes)}")
                continuation = self.generate_continuation(
                    cached_state,
                    suffix,
                    max_new_tokens
                )
                continuations.append(continuation)
                
            return continuations
            
        except Exception as e:
            logger.error(f"Error in batch_generate_continuations: {str(e)}")
            raise
