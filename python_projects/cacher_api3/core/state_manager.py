import uuid
import logging
from typing import Dict, Any, Optional
from core.model_loader import load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class StateManager:
    _instance = None

    def __new__(cls, device: str = "cuda"):
        if cls._instance is None:
            logger.info("Creating new StateManager instance")
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str = "cuda"):
        if not self._initialized:
            self.device = device
            self.states = {}  # In-memory state storage
            self.models = {}  # Model cache
            self.tokenizers = {}  # Tokenizer cache
            self._initialized = True
            logger.info(f"StateManager initialized with device: {device}")

    def create_state(self, model_id: str, text: str, config: Optional[Dict] = None) -> str:
        """Create and store a new state."""
        state_id = str(uuid.uuid4())
        
        # Load model and tokenizer if not cached
        if model_id not in self.models:
            logger.debug(f"Loading model and tokenizer for {model_id}")
            self.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            self.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)

        # Prepare initial state
        tokenizer = self.tokenizers[model_id]
        model = self.models[model_id]
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Store state
        self.states[state_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "config": config or {},
            "past_key_values": None
        }
        
        logger.info(f"State created with ID: {state_id}")
        logger.debug(f"State contents: {list(self.states[state_id].keys())}")
        return state_id

    def get_state(self, state_id: str) -> Optional[Dict]:
        """Retrieve a state by ID."""
        state = self.states.get(state_id)
        if not state:
            logger.warning(f"State with ID: {state_id} not found")
            return None
        logger.debug(f"Retrieved state {state_id} with keys: {list(state.keys())}")
        return state

    def update_state(self, state_id: str, updates: Dict) -> None:
        """Update an existing state."""
        if state_id not in self.states:
            logger.error(f"Cannot update non-existent state: {state_id}")
            raise ValueError(f"State {state_id} not found")
            
        current_state = self.states[state_id]
        current_state.update(updates)
        logger.debug(f"Updated state {state_id} with new keys: {list(updates.keys())}")

    def delete_state(self, state_id: str) -> bool:
        """Delete a state by its ID."""
        if state_id in self.states:
            del self.states[state_id]
            logger.info(f"State with ID: {state_id} deleted")
            return True
        else:
            logger.warning(f"State with ID: {state_id} not found for deletion")
            return False 