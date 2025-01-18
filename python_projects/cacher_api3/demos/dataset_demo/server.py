import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from dataclasses import dataclass
from queue import PriorityQueue
import logging
import signal
import sys
from contextlib import asynccontextmanager
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define static directory
STATIC_DIR = Path(__file__).parent / "static"

# Global flag for graceful shutdown
should_exit = False

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_CACHE: Dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForwardRequest(BaseModel):
    model_name: str
    text: str
    temperature: float = 1.0

class BeamSearchRequest(BaseModel):
    model_name: str
    text: str
    num_beams: int = 5
    max_length: int = 10
    temperature: float = 1.0

class TokenizeRequest(BaseModel):
    model_name: str
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None

class NextTokenRequest(BaseModel):
    model_name: str
    input_ids: List[int]
    temperature: float = 1.0

class ForwardSequenceRequest(BaseModel):
    model_name: str
    text: str
    max_tokens: int = 100
    temperature: float = 1.0

class AnalyzeSequenceRequest(BaseModel):
    model_name: str
    text: str
    temperature: float = 1.0

@dataclass
class BeamHypothesis:
    tokens: torch.Tensor
    score: float
    finished: bool = False

def get_model_and_tokenizer(model_name: str):
    if model_name not in MODEL_CACHE:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        MODEL_CACHE[model_name] = (model, tokenizer)
    return MODEL_CACHE[model_name]

@app.post("/api/forward")
async def forward(request: ForwardRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :] / request.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs[0], k=5)
            
            predictions = [
                {
                    "token": tokenizer.decode([idx.item()]),
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
        return {
            "predictions": predictions,
            "input_text": request.text
        }
        
    except Exception as e:
        logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/beam_search")
async def beam_search(request: BeamSearchRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        # Tokenize input
        input_ids = tokenizer(request.text, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        # Initialize beam search
        beams = [BeamHypothesis(input_ids[0], 0.0)]
        finished_beams = []
        
        # Perform beam search
        for step in range(request.max_length):
            if len(finished_beams) >= request.num_beams:
                break
                
            candidates = PriorityQueue()
            
            for beam_idx, beam in enumerate(beams):
                if beam.finished:
                    continue
                    
                with torch.no_grad():
                    outputs = model(beam.tokens.unsqueeze(0))
                    logits = outputs.logits[:, -1, :] / request.temperature
                    probs = F.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs[0], request.num_beams)
                    
                    for prob, token_id in zip(top_probs, top_indices):
                        new_tokens = torch.cat([beam.tokens, token_id.unsqueeze(0)])
                        score = beam.score - torch.log(prob).item()
                        
                        if token_id.item() == tokenizer.eos_token_id:
                            finished_beams.append((new_tokens, score))
                        else:
                            candidates.put((score, len(candidates.queue), BeamHypothesis(new_tokens, score)))
            
            beams = []
            for _ in range(min(request.num_beams - len(finished_beams), candidates.qsize())):
                score, _, hypothesis = candidates.get()
                beams.append(hypothesis)
        
        # Prepare results
        finished_beams.extend([(beam.tokens, beam.score) for beam in beams])
        finished_beams.sort(key=lambda x: x[1])
        
        results = []
        for tokens, score in finished_beams[:request.num_beams]:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append({"text": text, "score": score})
            
        return {"beams": results}
        
    except Exception as e:
        logger.error(f"Error in beam search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tokenize")
async def tokenize(request: TokenizeRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        if request.text is not None:
            # Tokenize text to ids
            inputs = tokenizer(request.text, return_tensors="pt")
            return {
                "input_ids": inputs.input_ids[0].tolist(),
                "tokens": [tokenizer.decode([id.item()]) for id in inputs.input_ids[0]],
                "text": request.text
            }
        elif request.input_ids is not None:
            # Decode ids to text
            text = tokenizer.decode(request.input_ids)
            return {
                "input_ids": request.input_ids,
                "tokens": [tokenizer.decode([id]) for id in request.input_ids],
                "text": text
            }
        else:
            raise HTTPException(status_code=400, detail="Either text or input_ids must be provided")
            
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/next_token")
async def next_token(request: NextTokenRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        # Convert input_ids to tensor
        input_ids = torch.tensor(request.input_ids).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits[:, -1, :] / request.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Get top k predictions
            top_k = 10
            top_probs, top_indices = torch.topk(probs[0], k=top_k)
            
            predictions = [
                {
                    "token": tokenizer.decode([idx.item()]),
                    "token_id": idx.item(),
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
        return {
            "predictions": predictions,
            "input_ids": input_ids.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in next token prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forward_sequence")
async def forward_sequence(request: ForwardSequenceRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        
        # Initial tokenization
        inputs = tokenizer(request.text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        sequence_stats = []
        current_ids = input_ids.clone()
        
        # Forward through the sequence
        for step in range(request.max_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[:, -1, :] / request.temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs[0], k=5)
                
                # Calculate entropy of the distribution
                entropy = -torch.sum(probs[0] * torch.log2(probs[0] + 1e-10))
                
                # Get next token
                next_token_id = top_indices[0].unsqueeze(0)
                next_token_prob = top_probs[0].item()
                
                # Add stats for this position
                sequence_stats.append({
                    "position": step + len(input_ids[0]),
                    "token": tokenizer.decode([next_token_id.item()]),
                    "token_id": next_token_id.item(),
                    "probability": next_token_prob,
                    "entropy": entropy.item(),
                    "top_predictions": [
                        {
                            "token": tokenizer.decode([idx.item()]),
                            "token_id": idx.item(),
                            "probability": prob.item()
                        }
                        for prob, idx in zip(top_probs, top_indices)
                    ],
                    "context": tokenizer.decode(current_ids[0][-10:])  # Last 10 tokens for context
                })
                
                # Append token to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Stop if we hit EOS
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        return {
            "original_text": request.text,
            "final_text": tokenizer.decode(current_ids[0]),
            "sequence_stats": sequence_stats
        }
        
    except Exception as e:
        logger.error(f"Error in forward sequence: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def handle_sigint(signum, frame):
    global should_exit
    should_exit = True
    logger.info("Received interrupt signal, finishing current request...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup/shutdown events"""
    signal.signal(signal.SIGINT, handle_sigint)
    yield
    logger.info("Shutting down...")
    model_names = list(MODEL_CACHE.keys())
    for model_name in model_names:
        logger.info(f"Unloading model: {model_name}")
        del MODEL_CACHE[model_name]

app = FastAPI(lifespan=lifespan)

@app.post("/api/analyze_sequence")
async def analyze_sequence(request: AnalyzeSequenceRequest):
    try:
        model, tokenizer = get_model_and_tokenizer(request.model_name)
        inputs = tokenizer(request.text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        sequence_stats = []
        
        for pos in range(input_ids.shape[1]):
            # Check if we should exit
            if should_exit:
                logger.info("Interrupting sequence analysis")
                break
                
            # Allow other coroutines to run
            await asyncio.sleep(0)
            
            current_ids = input_ids[:, :pos+1]
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[:, -1, :] / request.temperature
                probs = F.softmax(logits, dim=-1)
                
                actual_next_token = input_ids[0, pos].item()
                actual_next_token_prob = probs[0, actual_next_token].item()
                top_probs, top_indices = torch.topk(probs[0], k=5)
                entropy = -torch.sum(probs[0] * torch.log2(probs[0] + 1e-10))
                
                sequence_stats.append({
                    "position": pos,
                    "token": tokenizer.decode([input_ids[0, pos].item()]),
                    "token_id": input_ids[0, pos].item(),
                    "probability": actual_next_token_prob,
                    "entropy": entropy.item(),
                    "top_predictions": [
                        {
                            "token": tokenizer.decode([idx.item()]),
                            "token_id": idx.item(),
                            "probability": prob.item(),
                            "is_actual": idx.item() == actual_next_token
                        }
                        for prob, idx in zip(top_probs, top_indices)
                    ],
                    "context": tokenizer.decode(input_ids[0, max(0, pos-10):pos+1])
                })
        
        return {
            "text": request.text,
            "sequence_stats": sequence_stats,
            "interrupted": should_exit
        }
        
    except Exception as e:
        logger.error(f"Error in sequence analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = uvicorn.Config(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1
    )
    server = uvicorn.Server(config)
    server.run()



# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# DATASET_NAME = "O1-OPEN/OpenO1-SFT"
