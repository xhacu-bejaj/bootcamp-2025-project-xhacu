from pydantic import BaseModel, Field
from typing import Optional

from app.services.llm_client_factory import Provider

class PromptCreate(BaseModel):
    purpose: str = Field(..., examples=["summarize", "extract_entities"])
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

class PredictRequest(BaseModel):
    purpose: str
    document_text: str
    params: Optional[dict] = None
    #provider: str = "mock"
    provider: Provider = Provider.MOCK

class PredictResponse(BaseModel):
    output_text: str
    model_info: dict
    prompt_id: str
    prompt_version: int
    latency_ms: int
