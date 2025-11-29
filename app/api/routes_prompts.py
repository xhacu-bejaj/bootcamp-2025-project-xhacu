from fastapi import APIRouter, FastAPI, Header, HTTPException

from app.models.domain import Prompt
from app.models.schemas import PromptCreate, PromptRead, PromptPatch, PredictRequest, PredictResponse
<<<<<<< HEAD
from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId, Prompt
=======
from app.services.prompt_store import FileSnapshotStore, InMemoryStore, Purpose, UserId
>>>>>>> 5c33fb1d3439d4462d4e2d3f20da12a7270983f4
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
        new_prompt = store.create(
            purpose=data.purpose,
            name = data.name,
            template=data.template
        )
        return new_prompt

    
@router.get("/prompts/{purpose}", response_model=list[PromptRead])
def list_prompts(
<<<<<<< HEAD
        purpose: Purpose | None = None, # change to prompt
=======
        purpose: Purpose,
>>>>>>> 5c33fb1d3439d4462d4e2d3f20da12a7270983f4
        x_user_id: str = Header(default="user_anon")
    ):
        prompts_list = store.list(purpose)
        return prompts_list
<<<<<<< HEAD
    except Exception as e:
        raise e 

=======

    
>>>>>>> 5c33fb1d3439d4462d4e2d3f20da12a7270983f4
@router.patch("/prompts/{prompt_id}", response_model=PromptPatch)
def patch_prompt(
        prompt_id: str,
        data: PromptPatch,
        x_user_id: str = Header(default="user_anon")
    ):
<<<<<<< HEAD
    try:
        patch_prompt = store.patch(
            prompt_id=prompt_id,
            template=data.template
        )
        return patch_prompt
    except Exception as e:
        raise e
    
=======
        patched_prompt = store.patch(prompt_id=prompt_id,
                                     name=data.name,
                                     template=data.template)
        return patched_prompt

>>>>>>> 5c33fb1d3439d4462d4e2d3f20da12a7270983f4
@router.post("/prompts/{prompt_id}/activate")
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
    
<<<<<<< HEAD
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
=======
@router.get('/get_active/{purpose}')
def get_active(user_id: UserId, purpose: Purpose):
       return store.get_active(user_id=user_id, purpose=purpose)
    
@router.post("/predict", response_model=PredictResponse)
def predict_prompt(
>>>>>>> 5c33fb1d3439d4462d4e2d3f20da12a7270983f4
        req: PredictRequest,
        x_user_id: str = Header(default="user_anon"),
    ):
    ...