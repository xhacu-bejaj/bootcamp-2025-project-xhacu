from fastapi import APIRouter, Header, HTTPException, Depends, status

from app.models.schemas import PromptCreate, PromptRead, PromptPatch
from app.services.prompt_store import PromptStore, Purpose, UserId
from app.services.mongodb_store import MongoDBStore
from app.api.dependencies import get_store

from app.core.logging import setup_logging, log_api_call
from app.core.context import request_id_var

setup_logging()

prompt_router = APIRouter(prefix="/v1")

@prompt_router.get("/health")
@log_api_call
def health():
    return {"status": "ok", "request_id": request_id_var.get()}

@prompt_router.post("/prompts", response_model=PromptRead)
@log_api_call
async def create_prompt(
    data: PromptCreate,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
): 
    new_prompt = await store.create(
        purpose=data.purpose, name=data.name, template=data.template
    )
    return {**new_prompt.__dict__, "request_id": request_id_var.get()}

@prompt_router.get("/prompts/{purpose}", response_model=list[PromptRead])
@log_api_call
async def list_prompts(
    purpose: Purpose,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    prompts_list = await store.list(purpose)
    return [
        {
            **p.__dict__,
            "request_id": request_id_var.get(),
        }
        for p in prompts_list
    ]

@prompt_router.patch("/prompts/{prompt_id}", response_model=PromptRead)
@log_api_call
async def patch_prompt(
    prompt_id: str,
    data: PromptPatch,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    kwargs = {}
    if data.name is not None:
        kwargs["name"] = data.name
    if data.template is not None:
        kwargs["template"] = data.template
    
    patched_prompt = await store.patch(prompt_id=prompt_id, **kwargs)
    if patched_prompt:
        return {**patched_prompt.__dict__, "request_id": request_id_var.get()}
    return None

@prompt_router.post("/prompts/{prompt_id}/activate")
@log_api_call
async def activate_prompt(
    prompt_id: str,
    purpose: Purpose,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    active_prompt = await store.set_active(
        prompt_id=prompt_id, purpose=purpose, user_id=x_user_id
    )
    if active_prompt:
        return {**active_prompt.__dict__, "request_id": request_id_var.get()}
    return None

@prompt_router.get("/get_active/{purpose}")
@log_api_call
async def get_active(
    user_id: UserId, purpose: Purpose, store: PromptStore = Depends(get_store)
):
    active_prompt = await store.get_active(user_id=user_id, purpose=purpose)
    if active_prompt:
        return {**active_prompt.__dict__, "request_id": request_id_var.get()}
    return None

@prompt_router.post("/prompts/export")
@log_api_call
async def export_prompt_logs(
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    """Export prompt usage logs to CSV file.

    Triggers CSV export of all prompt predictions from MongoDB responses collection
    to /var/exports/prompt_logs.csv. Returns the path to the generated file.

    Returns:
        dict with export_path and total_records exported
    """

    if not isinstance(store, MongoDBStore):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export requires MongoDB store. Please configure MONGODB_URI in your environment.",
        )
    
    try:
        export_path = await store.export_prompt_usage_logs()
        return {
            "status": "success",
            "export_path": export_path,
            "message": "Prompt usage logs exported successfully",
            "request_id": request_id_var.get()
        }
    except IOError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during export: {str(e)}",
        )


