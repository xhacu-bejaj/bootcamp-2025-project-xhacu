from fastapi import APIRouter, Header, Depends

from app.services.processor import process_document
from app.models.schemas import PredictRequest, PredictResponse
from app.services.prompt_store import PromptStore
from app.api.dependencies import get_store
from app.services.mongodb_store import MongoDBStore
from app.core.logging import setup_logging, log_api_call
from app.core.context import request_id_var


setup_logging()
predict_router = APIRouter(prefix="/v1")


@predict_router.post("/predict", response_model=PredictResponse)
@log_api_call
async def predict_prompt(
    req: PredictRequest,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    purpose = req.purpose
    document_text = req.document_text
    provider = req.provider

    post_process_doc = await process_document(
        store, x_user_id, purpose, document_text, provider, params=req.params
    )

    # Store response only if using MongoDB store
    if isinstance(store, MongoDBStore):
        try:
            await store.store_response(post_process_doc, x_user_id, purpose)
        except Exception as e:
            print(f"Failed to store response: {e}")

    post_process_doc.request_id = request_id_var.get()
    return post_process_doc
