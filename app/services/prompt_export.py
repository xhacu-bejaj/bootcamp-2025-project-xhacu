import csv
import os
from pathlib import Path
from typing import Optional
from pymongo import MongoClient

from app.core.config import global_settings
from app.core.logging import log_api_call

class PromptExportService:
    """Service for exporting prompt usage logs to CSV."""

    def __init__(self, mongodb_uri: Optional[str] = None):
        """Initialize MongoDB connection for exporting logs.

        Args:
            mongodb_uri: Connection string for MongoDB
        """
        uri = mongodb_uri or global_settings.MONGODB_URI
        if not uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client["data"]
        self.responses_col = self.db["responses"]
        self.logs_col = self.db["logs"]

    @log_api_call
    def export_prompt_usage_logs(
        self, output_path: Optional[str] = None
    ) -> str:
        """Export prompt usage logs from responses collection to CSV.

        Args:
            output_path: Path where CSV file will be written.
                        If None, uses var/exports/prompt_logs.csv (relative to project root)

        Returns:
            Path to the created CSV file

        Raises:
            IOError: If CSV file cannot be created
        """

        if output_path is None:
            output_path = os.path.join("var", "exports", "prompt_logs.csv")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        documents = list(self.responses_col.find().sort("timestamp", -1))

        fieldnames = [
            "created_at",
            "prompt_id",
            "user_id",
            "purpose",
            "latency_ms",
            "model_info",
        ]

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for doc in documents:

                    model_info = doc.get("model_info", {})
                    model_info_str = f"{model_info.get('model', 'unknown')};temp={model_info.get('temperature', 'N/A')}"
                    
                    row = {
                        "created_at": doc.get("timestamp", "").isoformat()
                        if doc.get("timestamp")
                        else "",
                        "prompt_id": doc.get("prompt_id", ""),
                        "user_id": doc.get("user_id", ""),
                        "purpose": doc.get("purpose", ""),
                        "latency_ms": doc.get("latency_ms", 0),
                        "model_info": model_info_str,
                    }
                    writer.writerow(row)

            return output_path
        except IOError as e:
            raise IOError(f"Failed to write CSV file to {output_path}: {str(e)}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
