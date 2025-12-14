"""API routes for interacting with the AI agent."""
from fastapi import APIRouter, Header
from app.models.schemas import (
    AgentRequest, 
    AgentResponse, 
    TraceEventThought, 
    TraceEventToolCall, 
    TraceEventToolOutput,
    TraceEventFinalResponse
)
import sys
import os
from pathlib import Path
import uuid

# Add project root to path to import agent module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import root_agent
from app.core.logging import setup_logging, log_api_call
from app.core.config import settings
from app.core.context import request_id_var
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Set Google API key as environment variable for the ADK
os.environ['GOOGLE_API_KEY'] = settings.GOOGLE_API_KEY

setup_logging()
agent_router = APIRouter(prefix="/v1")

# Create a session service for the agent runner
session_service = InMemorySessionService()



@agent_router.post("/agent/query", response_model=AgentResponse, tags=["Agent"])
@log_api_call
async def query_agent(
    req: AgentRequest,
    x_user_id: str = Header(default="user_anon"),
):
    """
    Query the AI agent with access to the knowledge base.
    
    The agent can:
    - Search the ChromaDB knowledge base for relevant information
    - Answer questions based on stored documents
    - Provide the current date and time
    
    Args:
        req: Request containing the user's query, and optionally a session_id
        x_user_id: User identifier from header
        
    Returns:
        AgentResponse with the agent's answer and session_id
    """
    # Use user_id from request body if provided, otherwise use header
    user_id = req.user_id if req.user_id else x_user_id
    
    # Use existing session_id or create a new one for a new conversation
    session_id = req.session_id if req.session_id else str(uuid.uuid4())
    
    # Create a new session only if one doesn't already exist
    existing_session = await session_service.get_session(
        app_name="knowledge_base_agent", user_id=user_id, session_id=session_id
    )
    if not existing_session:
        await session_service.create_session(app_name="knowledge_base_agent", user_id=user_id, session_id=session_id)
    
    # Create the message content from the user query
    new_message = Content(
        role="user",
        parts=[Part(text=req.query)]
    )
    
    # Create a runner for the agent with session service and execute the query
    runner = Runner(
        app_name="knowledge_base_agent",
        agent=root_agent,
        session_service=session_service
    )
    
    # Collect all events from the async generator
    response_text = ""
    trace = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        # Safely check for and capture all known event types
        if thought := getattr(event, 'thought', None):
            trace.append(TraceEventThought(content=thought))

        if tool_code := getattr(event, 'tool_code', None):
            for tool_call in tool_code:
                trace.append(TraceEventToolCall(
                    tool_name=tool_call.name,
                    args=tool_call.args
                ))

        if tool_output := getattr(event, 'tool_output', None):
            for tool_out in tool_output:
                trace.append(TraceEventToolOutput(
                    tool_name=tool_out.tool_name,
                    output=str(tool_out.output) # Ensure output is a string
                ))

        if content := getattr(event, 'content', None):
            if content.parts:
                for part in content.parts:
                    if text := getattr(part, 'text', None):
                        response_text += text
                        trace.append(TraceEventFinalResponse(content=text))

    
    # If no response was collected, provide a default message
    if not response_text:
        response_text = "Agent completed but returned no response."
    
    return AgentResponse(
        response=response_text,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id_var.get() or "N/A",
        trace=trace
    )
