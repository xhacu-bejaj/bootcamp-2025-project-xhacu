from fastapi import APIRouter, HTTPException, Header

from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId
from app.services.processor import process_document
from app.core.config import settings
from app.models.domain import Prompt
from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
from app.services.llm_client_factory import LLMClientFactory
from app.api.routes_prompts import store
from app.core.logging import setup_logging, log_api_call


setup_logging()
predict_router = APIRouter(prefix="/v1")

@predict_router.post("/predict", response_model=PredictResponse)
@log_api_call
def predict_prompt(
        req: PredictRequest,
        x_user_id: str = Header(default="user_anon"),
    ):

    purpose=req.purpose
    document_text=req.document_text
    llm_params_dict = req.params.model_dump(exclude_none=True) if req.params else {}
    provider = req.provider

    post_process_doc = process_document(store, x_user_id, 
                                        purpose, 
                                        document_text, 
                                        provider, 
                                        **llm_params_dict)
    return post_process_doc

