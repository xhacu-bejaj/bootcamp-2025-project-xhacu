from fastapi import Request

from app.services.prompt_store import PromptStore
from app.services.chunk_store import ChunkStore
from app.core.context import app_context


def get_store(request: Request) -> PromptStore:
    """
    Dependency to get the store instance from the application state.
    """
    return request.app.state.prompt_store


def get_chunk_store() -> ChunkStore:
    """
    Dependency to get the shared chunk store instance from the app context.
    """
    if app_context.chunk_store is None:
        raise RuntimeError("ChunkStore has not been initialized.")
    return app_context.chunk_store
