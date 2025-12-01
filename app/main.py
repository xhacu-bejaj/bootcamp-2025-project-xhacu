from fastapi import FastAPI, Header, HTTPException

import uvicorn


from app.services.prompt_store import FileSnapshotStore, InMemoryStore
from app.services.processor import process_document
from app.core.errors import http_exception_handler, generic_exception_handler
from app.core.config import settings
from app.core.logging import setup_logging




#store = FileSnapshotStore("var/data.json") if settings.FILE_SNAPSHOT else InMemoryStore()
store = FileSnapshotStore() if settings.FILE_SNAPSHOT else InMemoryStore()
app = FastAPI(title="Prompted Doc Processor", version="0.1.0")

#app.add_exception_handler(HTTPException, http_exception_handler)

# Register the generic handler for all other Python Exceptions (5xx errors)
app.add_exception_handler(Exception, generic_exception_handler)
setup_logging()

from app.api.routes_prompts import prompt_router
from app.api.routes_predict import predict_router
app.include_router(prompt_router)
app.include_router(predict_router)


#if __name__ == "__main__":

    #uvicorn.run("main:app", host="127.0.0.1", port=8080, log_level="info", reload=True) 