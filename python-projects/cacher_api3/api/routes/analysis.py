import logging
import random
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

@router.post("/attention/{state_id}")
async def analyze_attention(state_id: str, request: AttentionRequest):
    """Analyze attention patterns for a state."""
    try:
        logger.info(f"Analyzing attention patterns for state {state_id}")
        
        # Verify state exists
        state = state_manager.get_state(state_id)
        
        # Generate mock attention patterns
        layer = request.layer or 0
        head = request.head or 0
        
        return {
            "attention_patterns": [
                {
                    "layer": layer,
                    "head": head,
                    "patterns": generate_mock_attention_pattern(
                        size=5,
                        threshold=request.threshold
                    )
                }
            ],
            "state_id": state_id
        }
    except ValueError as e:
        logger.error(f"State not found: {state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error analyzing attention: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/{state_id}")
async def inspect_cache(state_id: str, request: CacheInspectRequest):
    """Inspect key/value cache for a state."""
    try:
        logger.info(f"Inspecting cache for state {state_id}")
        
        # Verify state exists
        state = state_manager.get_state(state_id)
        
        # Mock cache data
        layer_range = request.layer_range or [0, 1]
        layers_data = []
        
        for layer in range(layer_range[0], layer_range[1] + 1):
            layers_data.append({
                "layer": layer,
                "keys": generate_mock_attention_pattern(3),
                "values": generate_mock_attention_pattern(3)
            })
        
        return {
            "cache_data": {
                "key_name": request.key_name or "all",
                "layers": layers_data
            },
            "state_id": state_id
        }
    except ValueError as e:
        logger.error(f"State not found: {state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error inspecting cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 