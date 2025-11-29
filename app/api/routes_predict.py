from fastapi import APIRouter, HTTPException, Header

from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId
from app.services.processor import process_document
from app.core.config import settings
from app.models.domain import Prompt
from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
from app.services.llm_client_factory import LLMClientFactory, Provider
from app.api.routes_prompts import store

predict_router = APIRouter(prefix="/v1")



@predict_router.post("/predict", response_model=PredictResponse)
def predict_prompt(
        req: PredictRequest,
        x_user_id: str = Header(default="user_anon"),
    ):
    purpose=req.purpose
    document_text=req.document_text
    params = {'provider': Provider.GOOGLE, 'temperature': 0.5}
    

    post_process_doc = process_document(store, x_user_id, purpose, document_text, **params)
    return post_process_doc

