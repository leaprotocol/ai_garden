import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from core import StateManager
from core.generation_utils import get_top_n_tokens
import torch

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize state manager
state_manager = StateManager(device="cpu")

class StateRequest(BaseModel):
    model_id: str
    text: Optional[str] = None
    state_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class StateResponse(BaseModel):
    state_id: str
    metadata: Dict[str, Any]

@router.post("", response_model=StateResponse)
async def create_state(request: StateRequest):
    """Create a new state for a model."""
    try:
        logger.info(f"Creating new state for model {request.model_id}")
        state_id = state_manager.create_state(
            model_id=request.model_id,
            text=request.text,
            config=request.config
        )
        # Get the full state data after creation
        state_data = state_manager.get_state(state_id)
        if not state_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve created state")
            
        # Prepare metadata excluding model and tokenizer objects
        metadata = {
            "model_id": request.model_id,
            "text": request.text,
            "config": request.config or {},
            "state_id": state_id
        }
        
        return StateResponse(
            state_id=state_id,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error creating state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{state_id}", response_model=StateResponse)
async def get_state(state_id: str):
    """Get the state by its ID."""
    state = state_manager.get_state(state_id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    return StateResponse(state_id=state_id, metadata=state)

@router.delete("/{state_id}", response_model=StateResponse)
async def delete_state(state_id: str):
    """Delete a state by its ID."""
    if state_manager.delete_state(state_id):
        return StateResponse(
            state_id=state_id,
            metadata={"message": f"State with ID: {state_id} deleted"}
        )
    raise HTTPException(status_code=404, detail="State not found")

@router.get("/{state_id}/top_tokens")
async def get_top_tokens(state_id: str, n: int = 5):
    """Get the top N tokens and their probabilities for the next token."""
    state = state_manager.get_state(state_id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    
    try:
        # Get the model's prediction for the next token
        with torch.no_grad():
            outputs = state["model"](state["input_ids"])
            next_token_logits = outputs.logits[:, -1, :]
            
        # Get top N tokens and their probabilities
        top_tokens = get_top_n_tokens(state["tokenizer"], next_token_logits[0], n=n)
        
        return {
            "tokens": [{"token": token, "probability": prob} for token, prob in top_tokens]
        }
    except Exception as e:
        logger.error(f"Error getting top tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 