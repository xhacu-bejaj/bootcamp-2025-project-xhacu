from fastapi import APIRouter, Header, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.services.db import get_session

from app.models.schemas import PromptCreate, PromptRead, PromptPatch
from app.services.prompt_store import PromptStore, Purpose, UserId
from app.services.mongodb_store import MongoDBStore
from app.api.dependencies import get_store

from app.core.logging import setup_logging, log_api_call


setup_logging()

prompt_router = APIRouter(prefix="/v1")


@prompt_router.get("/health")
@log_api_call
def health():
    return {"status": "ok"}


@prompt_router.post("/prompts", response_model=PromptCreate)
@log_api_call
async def create_prompt(
    data: PromptCreate,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):  # ->PromptRead:
    new_prompt = await store.create(
        purpose=data.purpose, name=data.name, template=data.template
    )
    return new_prompt


@prompt_router.get("/prompts/{purpose}", response_model=list[PromptRead])
@log_api_call
async def list_prompts(
    purpose: Purpose,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    prompts_list = await store.list(purpose)
    # Convert Prompt dataclass objects to dictionaries for Pydantic serialization
    return [
        {
            "id": p.id,
            "purpose": p.purpose,
            "name": p.name,
            "template": p.template,
            "version": p.version,
            "active": p.active,
        }
        for p in prompts_list
    ]


@prompt_router.patch("/prompts/{prompt_id}", response_model=PromptPatch)
@log_api_call
async def patch_prompt(
    prompt_id: str,
    data: PromptPatch,
    x_user_id: str = Header(default="user_anon"),
    store: PromptStore = Depends(get_store),
):
    patched_prompt = await store.patch(
        prompt_id=prompt_id, name=data.name, template=data.template
    )
    return patched_prompt


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
    return active_prompt


@prompt_router.get("/get_active/{purpose}")
@log_api_call
async def get_active(
    user_id: UserId, purpose: Purpose, store: PromptStore = Depends(get_store)
):
    return await store.get_active(user_id=user_id, purpose=purpose)


@prompt_router.get("/health/db", tags=["DatabaseSQLAlchemy"])
async def db_health_check(session: AsyncSession = Depends(get_session)):
    """
    Checks the database connection by executing a simple 'SELECT 1' query.
    If the query executes successfully, the database connection is healthy.
    """
    try:
        result = await session.execute(text("SELECT 1"))

        if result.scalar_one() == 1:
            return {
                "status": "ok",
                "message": "Database connection is healthy and operational.",
            }
        else:
            raise Exception("Database returned an unexpected result.")

    except Exception as e:
        print(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {e.__class__.__name__}",
        )


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
    # Check if using MongoDB store
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


