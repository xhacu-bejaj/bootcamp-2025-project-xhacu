from typing import List, Optional

from fastapi import APIRouter, Query

from app.models.schemas import HistoryItem
from app.api.routes_prompts import store
from app.services.mongodb_store import MongoDBStore
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

    if not isinstance(store, MongoDBStore):
        raise RuntimeError(
            "History retrieval requires MongoDB store. "
            "Please configure MONGODB_URI in your environment."
        )
    
    purpose_filter = purpose if purpose else None
    user_id_filter = user_id if user_id else None
    
    results = store.get_history(
        limit=limit, purpose=purpose_filter, user_id=user_id_filter
    )
  
    for item in results:
        if "timestamp" in item and hasattr(item["timestamp"], "isoformat"):
            item["timestamp"] = item["timestamp"].isoformat()
    return results
