import logging
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, AsyncIterator
from pydantic import BaseModel
from ...core import StateManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize state manager
state_manager = StateManager()

class TokenGenerationRequest(BaseModel):
    state_id: str
    max_tokens: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False

class BeamGenerationRequest(BaseModel):
    state_id: str
    num_beams: int = 5
    max_length: int = 50
    diversity_penalty: Optional[float] = None
    early_stopping: Optional[bool] = True

async def token_generator(request: TokenGenerationRequest) -> AsyncIterator[str]:
    """Generator function for streaming tokens."""
    try:
        # Get the state to check if it exists
        state = state_manager.get_state(request.state_id)
        
        # Mock token generation for demo
        words = ["continued", "on", "their", "journey", "through", "the", "digital", "landscape", "exploring", "new"]
        for i in range(min(request.max_tokens, len(words))):
            # Simulate some processing time
            await asyncio.sleep(0.2)
            yield f"{words[i]} "
    except Exception as e:
        logger.error(f"Error in token generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/token")
async def generate_token(request: TokenGenerationRequest):
    """Generate next token(s) for a given state."""
    try:
        logger.info(f"Generating tokens for state {request.state_id}")
        
        # Verify state exists
        state = state_manager.get_state(request.state_id)
        
        if request.stream:
            return StreamingResponse(
                token_generator(request),
                media_type="text/plain"
            )
        
        # For non-streaming, generate a single response
        words = ["adventure", "discovery", "journey", "exploration"]
        return {
            "tokens": words[:request.max_tokens],
            "state_id": request.state_id
        }
    except ValueError as e:
        logger.error(f"State not found: {request.state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error in token generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/beam")
async def generate_beam(request: BeamGenerationRequest):
    """Generate text using beam search."""
    try:
        logger.info(f"Starting beam search for state {request.state_id}")
        
        # Verify state exists
        state = state_manager.get_state(request.state_id)
        
        # Mock beam search results
        continuations = [
            ("continued their journey into the unknown", 0.95),
            ("ventured forth into the digital realm", 0.85),
            ("explored the vast landscape ahead", 0.75)
        ]
        
        return {
            "beams": [
                {"text": text, "score": score}
                for text, score in continuations[:request.num_beams]
            ],
            "state_id": request.state_id
        }
    except ValueError as e:
        logger.error(f"State not found: {request.state_id}")
        raise HTTPException(status_code=404, detail="State not found")
    except Exception as e:
        logger.error(f"Error in beam search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 