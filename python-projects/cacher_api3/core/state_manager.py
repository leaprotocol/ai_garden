import logging
from typing import Dict, Any, Optional, List
import uuid
import json
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.states: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized StateManager with cache directory: {cache_dir}")

    def create_state(self, model_id: str, text: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new state and return its ID."""
        state_id = str(uuid.uuid4())
        state = {
            "model_id": model_id,
            "text": text or "",
            "config": config or {},
            "metadata": {
                "created_at": str(datetime.now()),
                "last_modified": str(datetime.now())
            }
        }
        self.states[state_id] = state
        self._save_state(state_id)
        logger.info(f"Created new state {state_id} for model {model_id}")
        return state_id

    def get_state(self, state_id: str) -> Dict[str, Any]:
        """Get state by ID."""
        if state_id not in self.states:
            self._load_state(state_id)
        return self.states[state_id]

    def update_state(self, state_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with new data."""
        if state_id not in self.states:
            self._load_state(state_id)
        
        state = self.states[state_id]
        state.update(updates)
        state["metadata"]["last_modified"] = str(datetime.now())
        
        self._save_state(state_id)
        logger.info(f"Updated state {state_id}")
        return state

    def delete_state(self, state_id: str) -> None:
        """Delete state by ID."""
        if state_id in self.states:
            del self.states[state_id]
        
        state_path = self.cache_dir / f"{state_id}.json"
        if state_path.exists():
            state_path.unlink()
            logger.info(f"Deleted state {state_id}")

    def _save_state(self, state_id: str) -> None:
        """Save state to disk."""
        state_path = self.cache_dir / f"{state_id}.json"
        with state_path.open("w") as f:
            json.dump(self.states[state_id], f, indent=2)
        logger.debug(f"Saved state {state_id} to disk")

    def _load_state(self, state_id: str) -> None:
        """Load state from disk."""
        state_path = self.cache_dir / f"{state_id}.json"
        if not state_path.exists():
            raise ValueError(f"State {state_id} not found")
        
        with state_path.open("r") as f:
            self.states[state_id] = json.load(f)
        logger.debug(f"Loaded state {state_id} from disk")

    def list_states(self) -> List[str]:
        """List all available state IDs."""
        state_files = self.cache_dir.glob("*.json")
        return [f.stem for f in state_files]

    def fork_state(self, state_id: str) -> str:
        """Create a fork of an existing state."""
        original_state = self.get_state(state_id)
        new_state_id = str(uuid.uuid4())
        
        forked_state = {
            **original_state,
            "metadata": {
                **original_state["metadata"],
                "forked_from": state_id,
                "created_at": str(datetime.now()),
                "last_modified": str(datetime.now())
            }
        }
        
        self.states[new_state_id] = forked_state
        self._save_state(new_state_id)
        logger.info(f"Forked state {state_id} to {new_state_id}")
        return new_state_id 