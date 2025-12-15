from typing import Optional
from contextvars import ContextVar

from app.services.chunk_store import ChunkStore

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class AppContext:
    """
    A simple container for shared application state and dependencies.
    """
    chunk_store: Optional[ChunkStore] = None

# Create a single, globally accessible instance of the AppContext.
# This instance will be populated with dependencies during app startup.
app_context = AppContext()
