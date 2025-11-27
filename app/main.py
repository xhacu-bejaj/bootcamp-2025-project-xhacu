from fastapi import FastAPI, Header, HTTPException
import uvicorn


from app.services.prompt_store import FileSnapshotStore, InMemoryStore
from app.services.processor import process_document
from app.core.errors import http_error_handler
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes_prompts import router


#store = FileSnapshotStore("var/data.json") if settings.FILE_SNAPSHOT else InMemoryStore()
store = FileSnapshotStore() if settings.FILE_SNAPSHOT else InMemoryStore()
app = FastAPI(title="Prompted Doc Processor", version="0.1.0")
app.add_exception_handler(Exception, http_error_handler)
setup_logging()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, log_level="info", reload=True) 


