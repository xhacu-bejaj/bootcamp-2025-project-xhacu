"""API routes for interacting with the AI agent."""
from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from typing import Optional
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


class AgentRequest(BaseModel):
    """Request model for agent interaction."""
    query: str = Field(..., description="The question or prompt for the agent", examples=["What information do you have about machine learning?"])
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracking")


class AgentResponse(BaseModel):
    """Response model for agent interaction."""
    response: str = Field(..., description="The agent's response")
    user_id: str = Field(..., description="User identifier")
    request_id: Optional[str] = None


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
        req: Request containing the user's query
        x_user_id: User identifier from header
        
    Returns:
        AgentResponse with the agent's answer
    """
    # Use user_id from request body if provided, otherwise use header
    user_id = req.user_id if req.user_id else x_user_id
    
    # Generate a unique session_id for each request (stateless API calls)
    session_id = str(uuid.uuid4())
    
    # Create the session in the session service
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
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        # Extract text from event content if available
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text
    
    # If no response was collected, provide a default message
    if not response_text:
        response_text = "Agent completed but returned no response."
    
    return AgentResponse(
        response=response_text,
        user_id=user_id,
        request_id=request_id_var.get()
    )
