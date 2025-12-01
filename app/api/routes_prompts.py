from fastapi import APIRouter, FastAPI, Header, HTTPException


from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId

from app.core.config import settings
from app.core.logging import setup_logging, log_api_call



setup_logging()

prompt_router = APIRouter(prefix="/v1")
store = FileSnapshotStore() if settings.FILE_SNAPSHOT else InMemoryStore()


@prompt_router.get('/health')
@log_api_call
def health():
    return {'status':'ok'}

@prompt_router.post("/prompts", response_model=PromptCreate)
@log_api_call
def create_prompt(
        data: PromptCreate,
        x_user_id: str = Header(default="user_anon")
    ):
        new_prompt = store.create(
            purpose=data.purpose,
            name = data.name,
            template=data.template
        )
        return new_prompt

    
@prompt_router.get("/prompts/{purpose}", response_model=list[PromptRead])
@log_api_call
def list_prompts(
        purpose: Purpose,
        x_user_id: str = Header(default="user_anon")
    ):
        prompts_list = store.list(purpose)
        return prompts_list

    
@prompt_router.patch("/prompts/{prompt_id}", response_model=PromptPatch)
@log_api_call
def patch_prompt(
        prompt_id: str,
        data: PromptPatch,
        x_user_id: str = Header(default="user_anon")
    ):
        patched_prompt = store.patch(prompt_id=prompt_id,
                                     name=data.name,
                                     template=data.template)
        return patched_prompt

@prompt_router.post("/prompts/{prompt_id}/activate")
@log_api_call
def activate_prompt(
        prompt_id: str,
        purpose: Purpose,
        x_user_id: str = Header(default="user_anon"),
    ):
        active_prompt = store.set_active(
            prompt_id=prompt_id,
            purpose=purpose,
            user_id=x_user_id
        )
        return active_prompt
    
@prompt_router.get('/get_active/{purpose}')
@log_api_call
def get_active(user_id: UserId, purpose: Purpose):
       return store.get_active(user_id=user_id, purpose=purpose)
    
