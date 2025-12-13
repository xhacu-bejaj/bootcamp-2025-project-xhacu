from fastapi import Request

from app.services.prompt_store import PromptStore
from app.services.chunk_store import ChunkStore


def get_store(request: Request) -> PromptStore:
    """
    Dependency to get the store instance from the application state.
    """
    return request.app.state.prompt_store


def get_chunk_store(request: Request) -> ChunkStore:
    """
    Dependency to get the chunk store instance from the application state.
    """
    return request.app.state.chunk_store
