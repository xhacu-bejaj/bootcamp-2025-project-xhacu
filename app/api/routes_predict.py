from fastapi import APIRouter, HTTPException, Header

from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId
from app.services.processor import process_document
from app.core.config import settings
from app.models.domain import Prompt
from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
from app.services.llm_client_factory import LLMClientFactory, Provider

router = APIRouter(prefix="/v1")
store = FileSnapshotStore() if settings.FILE_SNAPSHOT else InMemoryStore()


@router.post("/predict", response_model=PredictResponse)
def predict_prompt(
        req: PredictRequest,
        x_user_id: str = Header(default="user_anon"),
    ):
    ...

