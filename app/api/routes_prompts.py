from fastapi import APIRouter, FastAPI, Header, HTTPException

from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId, Prompt
from app.services.processor import process_document
from app.core.errors import http_error_handler
from app.core.config import settings
from app.core.logging import setup_logging




router = APIRouter(prefix="/v1")
store = FileSnapshotStore() if settings.FILE_SNAPSHOT else InMemoryStore()


@router.get('/health')
def health():
    return {'status':'ok'}

@router.post("/prompts", response_model=PromptCreate)
def create_prompt(
        data: PromptCreate,
        x_user_id: str = Header(default="user_anon")
    ):
    try:
        new_prompt = store.create(
            purpose=data.purpose,
            name = data.name,
            template=data.template
        )
        return new_prompt
    except Exception as e:
        raise e
    
@router.get("/prompts/{purpose}", response_model=list[PromptRead])
def list_prompts(
        purpose: Purpose | None = None, # change to prompt
        x_user_id: str = Header(default="user_anon")
    ):
    try:
        prompts_list = store.list(purpose)
        return prompts_list
    except Exception as e:
        raise e 

@router.patch("/prompts/{prompt_id}", response_model=PromptPatch)
def patch_prompt(
        prompt_id: str,
        data: PromptPatch,
        x_user_id: str = Header(default="user_anon")
    ):
    try:
        patch_prompt = store.patch(
            prompt_id=prompt_id,
            template=data.template
        )
        return patch_prompt
    except Exception as e:
        raise e
    
@router.post("/prompts/{prompt_id}/activate")
def activate_prompt(
        prompt_id: str,
        purpose: Purpose,
        x_user_id: str = Header(default="user_anon"),
    ):
    try:
        active_prompt = store.set_active(
            prompt_id=prompt_id,
            purpose=purpose,
            user_id=x_user_id
        )
        return active_prompt
    except Exception as e:
        raise e
    
@router.get("/prompts/active")
def get_user_active_prompts(
                            user_id: UserId,
                            purpose: Purpose, 
                            #x_user_id: str = Header(default="user_anon")
                             ):
    try:
        active_prompt = store.get_active(user_id=user_id, purpose=purpose)
        return activate_prompt
    except Exception as e:
        raise e

    
@router.post("/predict", response_model=PredictResponse)
def predict(
        req: PredictRequest,
        x_user_id: str = Header(default="user_anon"),
    ):
    ...