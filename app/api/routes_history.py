from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.schemas import HistoryItem
from app.services.response_store import ResponseStore
from app.core.config import global_settings
from app.core.logging import log_api_call

history_router = APIRouter(prefix="/v1", tags=["history"])


@history_router.get("/history", response_model=List[HistoryItem])
@log_api_call
def get_history(
    limit: int = Query(default=50, ge=1, le=1000),
    purpose: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
):
    """Get recent prediction history from MongoDB.

    Args:
        limit: Maximum number of records to return (1-1000, default 50)
        purpose: Optional filter by purpose (e.g., 'summarize', 'translate')
        user_id: Optional filter by user_id

    Returns:
        List of recent predictions with timestamp, prompt_id, user_id, latency, provider/model
    """
    # Convert empty strings to None for proper filtering
    purpose_filter = purpose if purpose else None
    user_id_filter = user_id if user_id else None
    
    store = ResponseStore(mongodb_uri=global_settings.MONGODB_URI)
    try:
        results = store.get_history(
            limit=limit, purpose=purpose_filter, user_id=user_id_filter
        )
        # Convert datetime objects to ISO format strings
        for item in results:
            if "timestamp" in item and hasattr(item["timestamp"], "isoformat"):
                item["timestamp"] = item["timestamp"].isoformat()
        return results
    finally:
        store.close()
