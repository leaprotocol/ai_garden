import logging
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List, AsyncIterator
from pydantic import BaseModel
from core import StateManager
from core.generation_utils import generate_text
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize state manager
state_manager = StateManager(device="cpu")

class TokenGenerationRequest(BaseModel):
    state_id: str
    max_tokens: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None

class BeamGenerationRequest(BaseModel):
    state_id: str
    num_beams: int = 5
    max_length: int = 50
    diversity_penalty: Optional[float] = None
    early_stopping: Optional[bool] = True

    
@router.post("/token")
async def generate_token(request: TokenGenerationRequest):
    """Generate tokens based on the given state."""
    try:
        logger.info(f"Generating tokens for state: {request.state_id}")
        state = state_manager.get_state(request.state_id)
        logger.debug(f"Retrieved state from manager: {state}")
        
        if not state:
            logger.error(f"State not found: {request.state_id}")
            raise HTTPException(
                status_code=400,
                detail=f"State with ID {request.state_id} not found or expired"
            )

        # Verify required state components with detailed logging
        required_keys = ["model", "tokenizer", "text", "input_ids", "attention_mask"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            logger.error(f"State {request.state_id} missing components: {missing_keys}")
            logger.debug(f"Current state keys: {list(state.keys())}")
            raise HTTPException(
                status_code=422,
                detail=f"State is missing required components: {missing_keys}"
            )
        
        logger.debug(f"Generating with config: max_tokens={request.max_tokens}, temp={request.temperature}")
        outputs = generate_text(
            model=state["model"],
            tokenizer=state["tokenizer"],
            input_ids=state["input_ids"],
            attention_mask=state["attention_mask"],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            past_key_values=state.get("past_key_values"),
            use_cache=True
        )
        
        # Update and persist state with new outputs
        updated_state = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "past_key_values": outputs["past_key_values"],
            "text": outputs["text"]
        }
        logger.debug(f"Updating state with new values: {list(updated_state.keys())}")
        state_manager.update_state(request.state_id, updated_state)
        
        return {
            "text": outputs["text"],
            "state_id": request.state_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during token generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/beam")
async def generate_beam(request: BeamGenerationRequest):
    """Generate text using beam search."""
    state = state_manager.get_state(request.state_id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")

    model = state["model"]
    tokenizer = state["tokenizer"]
    input_ids = state["input_ids"]
    attention_mask = state["attention_mask"]

    try:
        generated_texts = model.beam_sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=request.num_beams,
            max_length=request.max_length,
            diversity_penalty=request.diversity_penalty,
            early_stopping=request.early_stopping
        )
        decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_texts]
        return {"texts": decoded_texts}
    except Exception as e:
        logger.error(f"Error during beam search generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 