import logging
import random
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from ...core import StateManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize state manager
state_manager = StateManager()

class AttentionRequest(BaseModel):
    state_id: str
    layer: Optional[int] = None
    threshold: Optional[float] = 0.1
    head: Optional[int] = None

class CacheInspectRequest(BaseModel):
    state_id: str
    key_name: Optional[str] = None
    layer_range: Optional[List[int]] = None

def generate_mock_attention_pattern(size: int = 5, threshold: float = 0.1) -> List[List[float]]:
    """Generate mock attention pattern matrix."""
    pattern = []
    for _ in range(size):
        row = [max(0.0, min(1.0, random.random())) for _ in range(size)]
        # Normalize row
        total = sum(row)
        row = [x/total for x in row]
        # Apply threshold
        row = [x if x > threshold else 0.0 for x in row]
        pattern.append(row)
    return pattern

@router.get("/tokens/{state_id}")
async def analyze_tokens(state_id: str):
    """Analyze token-level information for a state."""
    try:
        logger.info(f"Analyzing tokens for state {state_id}")
        
        # Verify state exists
        state = state_manager.get_state(state_id)
        text = state.get("text", "")
        
        # Mock token analysis
        tokens = text.split()
        return {
            "tokens": [
                {
                    "text": token,
                    "position": i,
                    "logprob": -random.uniform(1.0, 3.0)
                }
                for i, token in enumerate(tokens)
            ],
            "state_id": state_id
        }
    except ValueError as e:
        logger.error(f"State not found: {state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error analyzing tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/attention")
async def analyze_attention(request: AttentionRequest):
    """Analyze attention patterns for a specific state and layer."""
    try:
        logger.info(f"Analyzing attention for state {request.state_id}, layer {request.layer}, head {request.head}")
        # Mock attention analysis
        pattern = generate_mock_attention_pattern(threshold=request.threshold)
        return {"attention_pattern": pattern, "state_id": request.state_id, "layer": request.layer}
    except Exception as e:
        logger.error(f"Error analyzing attention: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/inspect")
async def inspect_cache(request: CacheInspectRequest):
    """Inspect the attention cache for a given state and layer."""
    try:
        logger.info(f"Inspecting cache for state {request.state_id}, key {request.key_name}, layers {request.layer_range}")
        # Mock cache inspection data
        cache_data = {"state_id": request.state_id, "key_name": request.key_name, "layer_range": request.layer_range, "data": {"mocked": True}}
        return cache_data
    except Exception as e:
        logger.error(f"Error inspecting cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 