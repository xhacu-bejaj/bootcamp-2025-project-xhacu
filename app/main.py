"""Main application entry point for Prompted Doc Processor.

Sets up FastAPI application with:
- Prompt storage (FileSnapshotStore, MongoDBStore, or InMemoryStore)
- API routers for prompts, predictions, database health, and history
- Exception handlers and logging
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import HTTPException

import uvicorn


from app.services.prompt_store import FileSnapshotStore, InMemoryStore
from app.services.mongodb_store import MongoDBStore
from app.services.chunk_store import ChunkStore
from app.core.errors import (
    generic_exception_handler,
    http_exception_handler,
    prompt_not_found_handler,
    llm_generation_handler,
    database_error_handler,
    configuration_error_handler,
)
from app.core.exceptions import (
    PromptNotFoundError,
    LLMGenerationError,
    DatabaseError,
    ConfigurationError,
)
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.context import app_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    if settings.FILE_SNAPSHOT:
        prompt_store = FileSnapshotStore()
    else:
        mongodb_uri = getattr(settings, "MONGODB_URI", None)
        if mongodb_uri and mongodb_uri.strip():
            try:
                prompt_store = MongoDBStore(mongodb_uri=mongodb_uri)
                await prompt_store.initialize()
            except Exception:
                prompt_store = InMemoryStore()
        else:
            prompt_store = InMemoryStore()
    
    app.state.prompt_store = prompt_store

    chunk_store = ChunkStore()
    chunk_store.initialize()
    app_context.chunk_store = chunk_store
    
    yield
    
    # Shutdown: close MongoDB connection if applicable
    if isinstance(app.state.prompt_store, MongoDBStore):
        await app.state.prompt_store.close()


from app.core.middleware import add_request_id_middleware
app = FastAPI(title="Prompted Doc Processor", version="0.1.0", lifespan=lifespan)
app.middleware("http")(add_request_id_middleware)

app.add_exception_handler(HTTPException, http_exception_handler) # type: ignore
app.add_exception_handler(PromptNotFoundError, prompt_not_found_handler) # type: ignore
app.add_exception_handler(LLMGenerationError, llm_generation_handler) # type: ignore
app.add_exception_handler(DatabaseError, database_error_handler) # type: ignore
app.add_exception_handler(ConfigurationError, configuration_error_handler) # type: ignore
app.add_exception_handler(Exception, generic_exception_handler)
setup_logging()

from app.api.routes_prompts import prompt_router  # noqa: E402
from app.api.routes_predict import predict_router  # noqa: E402
from app.api.routes_history import history_router  # noqa: E402
from app.api.routes_chunks import chunk_router  # noqa: E402
from app.api.routes_agent import agent_router  # noqa: E402

app.include_router(prompt_router)
app.include_router(predict_router)
app.include_router(history_router)
app.include_router(chunk_router)
app.include_router(agent_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, log_level="info", reload=True)
