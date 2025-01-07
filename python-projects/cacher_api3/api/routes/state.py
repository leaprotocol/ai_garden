import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from ...core import StateManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize state manager
state_manager = StateManager()

class StateRequest(BaseModel):
    model_id: str
    text: Optional[str] = None
    state_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class StateResponse(BaseModel):
    state_id: str
    metadata: Dict[str, Any]

@router.post("/", response_model=StateResponse)
async def create_state(request: StateRequest):
    """Create a new state for a model."""
    try:
        logger.info(f"Creating new state for model {request.model_id}")
        state_id = state_manager.create_state(
            model_id=request.model_id,
            text=request.text,
            config=request.config
        )
        state = state_manager.get_state(state_id)
        return StateResponse(
            state_id=state_id,
            metadata=state["metadata"]
        )
    except Exception as e:
        logger.error(f"Error creating state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{state_id}", response_model=StateResponse)
async def get_state(state_id: str):
    """Get state by ID."""
    try:
        logger.info(f"Retrieving state {state_id}")
        state = state_manager.get_state(state_id)
        return StateResponse(
            state_id=state_id,
            metadata=state["metadata"]
        )
    except ValueError as e:
        logger.error(f"State not found: {state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error retrieving state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{state_id}")
async def delete_state(state_id: str):
    """Delete state by ID."""
    try:
        logger.info(f"Deleting state {state_id}")
        state_manager.delete_state(state_id)
        return {"status": "deleted", "state_id": state_id}
    except Exception as e:
        logger.error(f"Error deleting state: {str(e)}")
        raise HTTPException(status_code=404, detail="State not found") 