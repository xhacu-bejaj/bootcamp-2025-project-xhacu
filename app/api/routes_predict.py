from fastapi import APIRouter, Header

from app.services.processor import process_document
from app.models.schemas import PredictRequest, PredictResponse
from app.api.routes_prompts import store
from app.services.mongodb_store import MongoDBStore
from app.core.logging import setup_logging, log_api_call


setup_logging()
predict_router = APIRouter(prefix="/v1")


@predict_router.post("/predict", response_model=PredictResponse)
@log_api_call
def predict_prompt(
    req: PredictRequest,
    x_user_id: str = Header(default="user_anon"),
):
    purpose = req.purpose
    document_text = req.document_text
    llm_params_dict = req.params.model_dump(exclude_none=True) if req.params else {}
    provider = req.provider

    post_process_doc = process_document(
        store, x_user_id, purpose, document_text, provider, **llm_params_dict
    )

    # Store response only if using MongoDB store
    if isinstance(store, MongoDBStore):
        try:
            store.store_response(post_process_doc, x_user_id, purpose)
        except Exception as e:
            print(f"Failed to store response: {e}")

    return post_process_doc
