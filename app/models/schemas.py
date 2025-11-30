from pydantic import BaseModel, Field
from typing import Optional

class PromptCreate(BaseModel):
    purpose: str = Field(..., examples=["summarize", "extract_entities", "translate"])
    name: str
    template: str

class PromptRead(BaseModel):
    id: str
    purpose: str
    name: str
    template: str
    version: int
    active: bool

class PromptPatch(BaseModel):
    name: Optional[str] = None
    template: Optional[str] = None

class LLMParams(BaseModel):
    temperature: Optional[float] = None
    #max_output_tokens: Optional[int] = None
    model_config = {
        "extra": "forbid"
    }

class PredictRequest(BaseModel):
    purpose: str
    document_text: str
    params: Optional[LLMParams] = None # Optional[dict] = None
    #provider: str = "mock"
    provider: str = 'google'

class ModelInfo(BaseModel):
    model: str
    temperature: float

class PredictResponse(BaseModel):
    output_text: str
    model_info: ModelInfo 
    prompt_id: str
    prompt_version: int
    latency_ms: int

class OpenaiOutputSchema(BaseModel):
    output_text: str
