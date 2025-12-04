from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pymongo import MongoClient
from app.core.config import global_settings
from app.models.schemas import PredictResponse
from app.core.logging import log_api_call


class ResponseStore:
    """Store LLM responses in MongoDB responses collection."""

    def __init__(self, mongodb_uri: Optional[str] = None):
        """Initialize MongoDB connection.

        Args:
            mongodb_uri: Connection string for MongoDB
        """
        uri = mongodb_uri or global_settings.MONGODB_URI
        if not uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client["data"]
        self.responses_col = self.db["responses"]

        # Create indexes for fast queries
        try:
            self.responses_col.create_index([("prompt_id", 1), ("timestamp", -1)])
            self.responses_col.create_index(
                [("user_id", 1), ("purpose", 1), ("timestamp", -1)]
            )
        except Exception:
            pass

    @log_api_call
    def store_response(
        self, response: PredictResponse, user_id: str, purpose: str
    ) -> None:
        """Store an LLM response as a document in responses collection.

        Args:
            response: PredictResponse object from LLM
            user_id: User who made the request
            purpose: Prompt purpose (summarize, translate, etc.)
        """
        doc = {
            "prompt_id": response.prompt_id,
            "user_id": user_id,
            "purpose": purpose,
            "output_text": response.output_text,
            "model_info": response.model_info.model_dump(),
            "latency_ms": response.latency_ms,
            "prompt_version": response.prompt_version,
            "timestamp": datetime.now(timezone.utc),
        }

        self.responses_col.insert_one(doc)

    @log_api_call
    def get_history(
        self, limit: int = 50, purpose: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent predictions from MongoDB.

        Args:
            limit: Maximum number of records to return (default 50)
            purpose: Optional filter by purpose
            user_id: Optional filter by user_id

        Returns:
            List of prediction records with timestamp, prompt_id, user_id, latency, provider/model
        """
        query = {}
        if purpose:
            query["purpose"] = purpose
        if user_id:
            query["user_id"] = user_id

        cursor = (
            self.responses_col.find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )

        results = []
        for doc in cursor:
            results.append(
                {
                    "timestamp": doc["timestamp"],
                    "prompt_id": doc["prompt_id"],
                    "user_id": doc["user_id"],
                    "purpose": doc["purpose"],
                    "latency_ms": doc["latency_ms"],
                    "provider": doc["model_info"].get("model", "unknown"),
                    "model": doc["model_info"].get("model", "unknown"),
                    "prompt_version": doc.get("prompt_version", 1),
                }
            )
        return results

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
