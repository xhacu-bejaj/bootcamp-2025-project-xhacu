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
    model_config = {"extra": "forbid"}


class PredictRequest(BaseModel):
    purpose: str
    document_text: str
    params: Optional[LLMParams] = None 
    provider: str = "google"


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


class HistoryItem(BaseModel):
    timestamp: str
    prompt_id: str
    user_id: str
    purpose: str
    latency_ms: int
    provider: str
    model: str
    prompt_version: int


class ChunkMetadataInput(BaseModel):
    """Schema for chunk metadata input"""
    source: Optional[str] = Field(None, description="Source document or identifier")
    position: Optional[int] = Field(None, description="Position in the original document")


class ChunkMetadata(BaseModel):
    """Schema for chunk metadata response"""
    length: int = Field(..., description="Length of the chunk text in characters")
    source: Optional[str] = Field(None, description="Source document or identifier")
    position: Optional[int] = Field(None, description="Position in the original document")


class ChunkInsert(BaseModel):
    """Schema for inserting a chunk into the vector database."""
    text: str = Field(..., description="The text content of the chunk")
    metadata: Optional[ChunkMetadataInput] = Field(None, description="Metadata associated with the chunk")


class ChunkResponse(BaseModel):
    """Schema for chunk retrieval response."""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata associated with the chunk")
    distance: Optional[float] = Field(None, description="Distance/similarity score from query")


class ChunkRetrieveRequest(BaseModel):
    """Schema for chunk retrieval request."""
    text: str = Field(..., description="Query text to find similar chunks")
    n_chunks: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")
