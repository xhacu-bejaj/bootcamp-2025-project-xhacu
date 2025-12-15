import logging
import asyncio
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.core.context import request_id_var 

class AsyncMongoDBHandler(logging.Handler):
    """
    Custom logging handler that writes logs to MongoDB asynchronously.
    """
    _client = None
    _logs_col = None

    def __init__(self, mongodb_uri: str, db_name: str = "data", collection_name: str = "logs"):
        super().__init__()
        self.uri = mongodb_uri or settings.MONGODB_URI
        self.db_name = db_name
        self.collection_name = collection_name
        if not self.uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

    def _get_collection(self):
        if self._logs_col is None:
            try:
                self._client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
                db = self._client[self.db_name]
                self._logs_col = db[self.collection_name]
            except Exception as e:
                import sys
                print(f"CRITICAL: Could not create MongoDB client for logging: {e}", file=sys.stderr)
                return None
        return self._logs_col

    def emit(self, record):
        """
        Format the log record and schedule its insertion on the event loop.
        """
        collection = self._get_collection()
        if collection is None:
            return 

        log_doc = {
            "timestamp": datetime.now(timezone.utc),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": request_id_var.get("N/A")
        }
        if record.exc_info:
            log_doc["exception"] = self.format(record)

        async def do_insert():
            try:
                await collection.insert_one(log_doc)
            except Exception as e:
                import sys
                print(f"Failed to write log to MongoDB: {e}", file=sys.stderr)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(do_insert())
        except RuntimeError:
            pass

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
        super().close()
