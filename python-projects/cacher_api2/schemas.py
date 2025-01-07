from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ForwardRequest(BaseModel):
    text: str
    cache_id: Optional[str] = None

class ForwardResponse(BaseModel):
    cache_id: str
    input_length: int

class NextTokenProbsRequest(BaseModel):
    cache_id: str

class NextTokenProbsResponse(BaseModel):
    probabilities: Dict[str, float]

class BeamSearchRequest(BaseModel):
    cache_id: str
    beam_width: int = Field(5, ge=1, description="Beam width for beam search")
    max_length: int = Field(50, ge=1, description="Maximum length of generated sequence")

class BeamSearchResponse(BaseModel):
    sequences: List[str]

class GreedyTokenRequest(BaseModel):
    cache_id: str

class GreedyTokenResponse(BaseModel):
    token: str
    probability: float

class SaveCacheRequest(BaseModel):
    cache_id: str
    filename: str

class SaveCacheResponse(BaseModel):
    success: bool
    filepath: str

class LoadCacheRequest(BaseModel):
    filename: str

class LoadCacheResponse(BaseModel):
    cache_id: str

class ListCachesResponse(BaseModel):
    caches: Dict[str, int]

class CacheSizeRequest(BaseModel):
    cache_id: str

class CacheSizeResponse(BaseModel):
    cache_size: int

class StreamGenerationRequest(BaseModel):
    cache_id: str
    max_length: int = Field(50, ge=1, description="Maximum length of generated sequence")
    
class StreamGenerationResponse(BaseModel):
    token: str
    step_number: int
    partial_text: str 