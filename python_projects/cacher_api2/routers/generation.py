from fastapi import APIRouter, HTTPException, WebSocket, Query, Depends
from cacher_api2.services.generation_service import GenerationService
from cacher_api2.schemas import *
from cacher_api2.utils import get_logger
from typing import Annotated
from cacher_api2.dependencies import get_cache_service, get_generation_service

router = APIRouter()
logger = get_logger(__name__)

@router.post("/forward/{model_id}", response_model=ForwardResponse)
async def forward(
    model_id: str,
    request: ForwardRequest,
    generation_service: Annotated[GenerationService, Depends(get_generation_service)]
):
    """
    Performs a forward pass through the specified model.
    """
    logger.debug(f"POST /forward/{model_id} called with body: {request.json()}")
    try:
        cache_id, input_length = await generation_service.forward(model_id, request.text, request.cache_id)
        return ForwardResponse(cache_id=cache_id, input_length=input_length)
    except ValueError as e:
        logger.error(f"ValueError in forward: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in forward: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/next_token_probs/{model_id}/{cache_id}", response_model=NextTokenProbsResponse)
async def get_next_token_probs(
    model_id: str,
    cache_id: str,
    generation_service: Annotated[GenerationService, Depends(get_generation_service)]
):
    """
    Gets the probabilities of the next tokens given a cache ID.
    """
    logger.debug(f"GET /next_token_probs/{model_id}/{cache_id} called")
    try:
        probabilities = await generation_service.get_next_token_probs(model_id, cache_id)
        return NextTokenProbsResponse(probabilities=probabilities)
    except ValueError as e:
        logger.error(f"ValueError in get_next_token_probs: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_next_token_probs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/beam_search/{model_id}/{cache_id}", response_model=BeamSearchResponse)
async def beam_search(
    model_id: str,
    cache_id: str,
    generation_service: Annotated[GenerationService, Depends(get_generation_service)],
    beam_width: int = Query(5, ge=1, description="Beam width for beam search"),
    max_length: int = Query(50, ge=1, description="Maximum length of generated sequence")
):
    """
    Performs beam search from a given cache state.
    """
    logger.debug(f"GET /beam_search/{model_id}/{cache_id} called with beam_width={beam_width}, max_length={max_length}")
    try:
        sequences = await generation_service.beam_search(model_id, cache_id, beam_width, max_length)
        return BeamSearchResponse(sequences=sequences)
    except ValueError as e:
        logger.error(f"ValueError in beam_search: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in beam_search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/greedy_token/{model_id}/{cache_id}", response_model=GreedyTokenResponse)
async def get_greedy_token(
    model_id: str,
    cache_id: str,
    generation_service: Annotated[GenerationService, Depends(get_generation_service)]
):
    """
    Gets the most likely next token (greedily) given a cache ID.
    """
    logger.debug(f"GET /greedy_token/{model_id}/{cache_id} called")
    try:
        token, probability = await generation_service.get_greedy_token(model_id, cache_id)
        return GreedyTokenResponse(token=token, probability=probability)
    except ValueError as e:
        logger.error(f"ValueError in get_greedy_token: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_greedy_token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.websocket("/generate_stream/{model_id}/{cache_id}")
async def generate_stream(
    websocket: WebSocket,
    model_id: str,
    cache_id: str,
    generation_service: Annotated[GenerationService, Depends(get_generation_service)],
    max_length: int = Query(50, ge=1, description="Maximum length of generated sequence")
):
    """
    Streams generated text over a WebSocket connection.
    """
    await websocket.accept()
    logger.debug(f"WebSocket /generate_stream/{model_id}/{cache_id} connected")

    model: PreTrainedModel = generation_service.model_manager.get_model(model_id)
    tokenizer: PreTrainedTokenizer = generation_service.model_manager.get_tokenizer(model_id)

    if cache_id not in generation_service.cache_service.cached_states:
        logger.error(f"Cache not found: {cache_id}")
        await websocket.send_json({"error": "Cache not found"})
        await websocket.close()
        return

    cache = generation_service.cache_service.cached_states[cache_id]
    input_ids = cache['input_ids']
    attention_mask = cache['attention_mask']
    past_key_values = cache['past_key_values']
    
    step_number = 0
    partial_text = ""

    try:
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            next_token_text = tokenizer.decode(next_token)
            
            # Update input_ids, attention_mask, and past_key_values for next iteration
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=model.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=model.device)], dim=-1)
            past_key_values = outputs.past_key_values

            # Check for end of sequence
            if next_token == tokenizer.eos_token_id:
                break
            
            step_number += 1
            partial_text += next_token_text

            await websocket.send_json(
                StreamGenerationResponse(
                    token=next_token_text,
                    step_number=step_number,
                    partial_text=partial_text
                ).dict()
            )

    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        await websocket.send_json({"error": "Error during generation"})
    finally:
        await websocket.close()
        logger.debug(f"WebSocket /generate_stream/{model_id}/{cache_id} closed") 