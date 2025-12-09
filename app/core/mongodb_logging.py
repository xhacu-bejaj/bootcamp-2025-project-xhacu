import logging
from datetime import datetime, timezone
from pymongo import MongoClient
from app.core.config import settings


class MongoDBHandler(logging.Handler):
    """Custom logging handler that writes logs to MongoDB."""

    def __init__(
        self, mongodb_uri: str, db_name: str = "data", collection_name: str = "logs"
    ):
        """Initialize MongoDB logging handler.
        
        Args:
            mongodb_uri: MongoDB connection string
            db_name: Database name (default: 'data')
            collection_name: Collection name for logs (default: 'logs')
            
        Raises:
            ValueError: If mongodb_uri is not provided
        """
        super().__init__()
        uri = mongodb_uri or settings.MONGODB_URI
        if not uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.logs_col = self.db[collection_name]

        # Create index for timestamp for easy querying and TTL
        try:
            self.logs_col.create_index("timestamp")
            self.logs_col.create_index([("level", 1), ("timestamp", -1)])
        except Exception:
            pass

    def emit(self, record):
        """Write log record to MongoDB.
        
        Creates a document with timestamp, level, logger name, message,
        and source code location. Includes exception info if present.
        
        Args:
            record: LogRecord to write to MongoDB
        """
        try:
            log_doc = {
                "timestamp": datetime.now(timezone.utc),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname,
            }

            # Add exception info if present
            if record.exc_info:
                log_doc["exception"] = self.format(record)

            self.logs_col.insert_one(log_doc)
        except Exception:
            # Silently fail if logging to MongoDB fails
            self.handleError(record)

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        super().close()
