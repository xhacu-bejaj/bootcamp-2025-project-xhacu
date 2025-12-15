from typing import Optional
from contextvars import ContextVar

from app.services.chunk_store import ChunkStore

class AppContext:
    """
    A simple container for shared application state and dependencies.
    """
    chunk_store: Optional[ChunkStore] = None
    
app_context = AppContext()
