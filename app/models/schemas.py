from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Union, Literal


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
    request_id: Optional[str] = None

class PromptPatch(BaseModel):
    name: Optional[str] = None
    template: Optional[str] = None

class LLMParams(BaseModel):
    model: str 
    temperature: float
    model_config = {"extra": "forbid"}

class PredictRequest(BaseModel):
    purpose: str
    document_text: str
    params: Optional[LLMParams] = None  
    provider: str = "google"

class PredictResponse(BaseModel):
    output_text: str
    model_info: LLMParams
    prompt_id: str
    prompt_version: int
    latency_ms: int
    request_id: Optional[str] = None

class LLMOutput(BaseModel):
    output_text: str
    latency_ms: int
    model_info: LLMParams

class OutputSchema(BaseModel):
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
    request_id: Optional[str] = None

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


class ChunkInsertBatch(BaseModel):
    """Schema for inserting a batch of chunks."""
    chunks: List[ChunkInsert]


class ChunkResponse(BaseModel):
    """Schema for chunk retrieval response."""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata associated with the chunk")
    distance: Optional[float] = Field(None, description="Distance/similarity score from query")
    request_id: Optional[str] = None


class ChunkRetrieveRequest(BaseModel):
    """Schema for chunk retrieval request."""
    text: str = Field(..., description="Query text to find similar chunks")
    n_chunks: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")


class AgentRequest(BaseModel):
    """Request model for agent interaction."""
    query: str = Field(..., description="The question or prompt for the agent")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracking")
    session_id: Optional[str] = Field(None, description="Optional session identifier for conversational context")


# --- Trace Event Models ---

class TraceEventThought(BaseModel):
    """Represents the agent's reasoning or thought process."""
    type: Literal["thought"] = "thought"
    content: str = Field(..., description="The text of the agent's thought.")

class TraceEventToolCall(BaseModel):
    """Represents the agent's decision to call a tool."""
    type: Literal["tool_code"] = "tool_code"
    tool_name: str = Field(..., description="The name of the tool being called.")
    args: Dict[str, Any] = Field(..., description="The arguments passed to the tool.")

class TraceEventToolOutput(BaseModel):
    """Represents the output received from a tool execution."""
    type: Literal["tool_output"] = "tool_output"
    tool_name: str = Field(..., description="The name of the tool that was called.")
    output: str = Field(..., description="The string output returned by the tool.")

class TraceEventFinalResponse(BaseModel):
    """Represents a part of the agent's final response to the user."""
    type: Literal["response"] = "response"
    content: str = Field(..., description="A chunk of the final response text.")

TraceEvent = Union[TraceEventThought, TraceEventToolCall, TraceEventToolOutput, TraceEventFinalResponse]

class AgentResponse(BaseModel):
    """Response model for agent interaction."""
    response: str = Field(..., description="The agent's response")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier for conversational context")
    request_id: str
    trace: List[TraceEvent] = Field(..., description="A detailed trace of the agent's execution steps")
