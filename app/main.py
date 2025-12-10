"""Main application entry point for Prompted Doc Processor.

Sets up FastAPI application with:
- Prompt storage (FileSnapshotStore, MongoDBStore, or InMemoryStore)
- API routers for prompts, predictions, database health, and history
- Exception handlers and logging
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

import uvicorn


from app.services.prompt_store import FileSnapshotStore, InMemoryStore
from app.services.mongodb_store import MongoDBStore
from app.core.errors import generic_exception_handler
from app.core.config import settings
from app.core.logging import setup_logging


if settings.FILE_SNAPSHOT:
    store = FileSnapshotStore()
else:
    mongodb_uri = getattr(settings, "MONGODB_URI", None)
    if mongodb_uri and mongodb_uri.strip():
        try:
            store = MongoDBStore(mongodb_uri=mongodb_uri)
        except Exception:
            store = InMemoryStore()
    else:
        store = InMemoryStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup: store is already initialized above
    yield
    # Shutdown: close MongoDB connection if applicable
    if isinstance(store, MongoDBStore):
        store.close()


app = FastAPI(title="Prompted Doc Processor", version="0.1.0", lifespan=lifespan)

# app.add_exception_handler(HTTPException, http_exception_handler)

app.add_exception_handler(Exception, generic_exception_handler)
setup_logging()

from app.api.routes_prompts import prompt_router  # noqa: E402
from app.api.routes_predict import predict_router  # noqa: E402
from app.api.routes_db import db_router  # noqa: E402
from app.api.routes_history import history_router  # noqa: E402
from app.api.routes_chunks import chunk_router  # noqa: E402

app.include_router(prompt_router)
app.include_router(predict_router)
app.include_router(db_router)
app.include_router(history_router)
app.include_router(chunk_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, log_level="info", reload=True)
