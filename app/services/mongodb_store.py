from typing import Optional, TypeAlias, List, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone
import csv
import os
from pathlib import Path
import asyncio


from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

from ..models.domain import Prompt
from app.models.schemas import PredictResponse
from app.core.logging import log_service_call
from app.core.context import request_id_var
from app.core.config import settings
from app.services.prompt_store import PromptStore
from app.core.exceptions import PromptNotFoundError 

UserId: TypeAlias = str
PromptId: TypeAlias = str
Purpose: TypeAlias = str

class MongoDBStore(PromptStore):
    """MongoDB-backed implementation of PromptStore interface."""
    @log_service_call
    def __init__(self, mongodb_uri: Optional[str] = None):
        """Initialize MongoDB connection and setup collections.

        Args:
            mongodb_uri: Connection string for MongoDB. Defaults to settings.MONGODB_URI

        Raises:
            ValueError: If MONGODB_URI is not provided and not in settings
            ConnectionFailure: If unable to connect to MongoDB
        """
        uri = mongodb_uri or settings.MONGODB_URI
        if not uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

        try:
            self.client: AsyncIOMotorClient = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB at {uri}: {e}")

        db_name = getattr(settings, "MONGODB_DB_NAME", "data")
        prompts_col_name = getattr(
            settings, "MONGODB_PROMPTS_COLLECTION", "prompts_db"
        )
        active_col_name = getattr(
            settings, "MONGODB_ACTIVE_COLLECTION", "active_prompts"
        )
        self.db = self.client[db_name]
        self.prompts_collection = self.db[prompts_col_name]
        self.active_prompts_collection = self.db[active_col_name]
        self.responses_collection = self.db["responses"]

    async def initialize(self):
        """Run async-only initialization tasks."""
        await self.client.admin.command("ping")
        await self._create_indexes()

    async def _create_indexes(self):
        """Create indexes for common queries."""
        try:
            await self.prompts_collection.create_index("purpose")
            await self.prompts_collection.create_index([("purpose", 1), ("version", -1)])
        except Exception:
            pass

        try:
            await self.active_prompts_collection.create_index(
                [("user_id", 1), ("purpose", 1)], unique=True
            )
        except Exception:
            pass

        try:
            await self.responses_collection.create_index([("prompt_id", 1), ("timestamp", -1)])
            await self.responses_collection.create_index(
                [("user_id", 1), ("purpose", 1), ("timestamp", -1)]
            )
        except Exception:
            pass

    def _doc_to_prompt(self, doc: dict) -> Prompt: 
        """Convert MongoDB document to Prompt domain object."""
        return Prompt(
            id=str(doc["_id"]), 
            purpose=doc["purpose"],
            name=doc.get("name", "Unnamed"),  
            template=doc.get("template", ""), 
            version=doc.get("version", 1), 
            active=doc.get("active", False),
            request_id=doc.get("request_id"),
        )

    def _prompt_to_doc(self, prompt: Prompt) -> dict:
        """Convert Prompt domain object to MongoDB document."""
        return {
            "_id": prompt.id, 
            "purpose": prompt.purpose,
            "name": prompt.name,
            "template": prompt.template,
            "version": prompt.version,
            "active": prompt.active,
            "request_id": prompt.request_id,
        }
    
    @log_service_call
    async def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
        """Create a new prompt.

        Args:
            purpose: Category/purpose of the prompt
            name: Human-readable name
            template: Jinja2 template string

        Returns:
            Created Prompt object
        """
        prompt = Prompt(
            id=str(uuid4()),
            purpose=purpose,
            name=name,
            template=template,
            version=1,
            active=False,
            request_id=request_id_var.get(),
        )

        doc = self._prompt_to_doc(prompt)
        await self.prompts_collection.insert_one(doc)
        return prompt

    @log_service_call
    async def list(self, purpose: Purpose | None = None) -> list[Prompt]:
        """List all prompts for a given purpose.

        Args:
            purpose: Filter prompts by this purpose. If None, returns all prompts.

        Returns:
            List of Prompt objects
        """
        query = {"purpose": purpose} if purpose is not None else {}
        cursor = self.prompts_collection.find(query)
        docs = await cursor.to_list(length=None)
        return [self._doc_to_prompt(doc) for doc in docs] 

    @log_service_call
    async def get(self, prompt_id: PromptId) -> Prompt:
        """Retrieve a prompt by ID.

        Args:
            prompt_id: ID of the prompt to retrieve

        Returns:
            Prompt object

        Raises:
            PromptNotFoundError: If the prompt with the given ID is not found.
        """
        doc = await self.prompts_collection.find_one({"_id": prompt_id})
        if not doc:
            raise PromptNotFoundError(f"Prompt with ID {prompt_id} not found")
        return self._doc_to_prompt(doc)

    @log_service_call
    async def delete(self, prompt_id: PromptId) -> bool:
        """Delete a prompt by ID.

        Args:
            prompt_id: ID of the prompt to delete.

        Returns:
            True if the prompt was deleted, False otherwise.
        """
        
        await self.active_prompts_collection.delete_many({"prompt_id": prompt_id})
        result = await self.prompts_collection.delete_one({"_id": prompt_id})
        return result.deleted_count > 0

    @log_service_call
    async def patch(
        self, prompt_id: PromptId, name: str | None = None, template: str | None = None
    ) -> Prompt | None:
        """Update a prompt's name and/or template.

        Args:
            prompt_id: ID of the prompt to update
            name: New name (optional)
            template: New template (optional)

        Returns:
            Updated Prompt object or None if not found
        """
        update_dict = {}

        if name is not None:
            update_dict["name"] = name

        if template is not None:
            update_dict["template"] = template

        if update_dict:
            result = await self.prompts_collection.find_one_and_update(
                {"_id": prompt_id},
                {"$set": update_dict, "$inc": {"version": 1}},
                return_document=True,
            )

            return self._doc_to_prompt(result) if result else None

        return await self.get(prompt_id)
    
    @log_service_call
    async def set_active(
        self, user_id: UserId, purpose: Purpose, prompt_id: PromptId
    ) -> Prompt | None:
        """Set a prompt as active for a user and purpose.

        Args:
            user_id: User identifier
            purpose: Prompt purpose/category
            prompt_id: ID of prompt to activate

        Returns:
            The activated Prompt object or None if prompt not found
        """
        prompt = await self.get(prompt_id)
        if not prompt:
            return None

        active_doc = await self.active_prompts_collection.find_one(
            {"user_id": user_id, "purpose": purpose}
        )

        if active_doc:
            old_prompt_id = active_doc["prompt_id"]
            await self.prompts_collection.update_one(
                {"_id": old_prompt_id}, {"$set": {"active": False}}
            )

        await self.active_prompts_collection.update_one(
            {"user_id": user_id, "purpose": purpose},
            {"$set": {"prompt_id": prompt_id}},
            upsert=True,
        )

        await self.prompts_collection.update_one(
            {"_id": prompt_id}, {"$set": {"active": True}}
        )

        return await self.get(prompt_id)
    
    @log_service_call    
    async def get_active(self, user_id: UserId, purpose: Purpose) -> Prompt | None:
        """Retrieve the active prompt for a user and purpose.

        Args:
            user_id: User identifier
            purpose: Prompt purpose/category

        Returns:
            Active Prompt object or None if not set
        """
        active_doc = await self.active_prompts_collection.find_one(
            {"user_id": user_id, "purpose": purpose}
        )

        if not active_doc:
            return None

        return await self.get(active_doc["prompt_id"])

    @log_service_call
    async def store_response(
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

        await self.responses_collection.insert_one(doc)

    @log_service_call
    async def get_history(
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
            self.responses_collection.find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )

        results = []
        async for doc in cursor:
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
    
    @log_service_call
    async def _export_to_csv(self, output_path: str):
        """Export documents to CSV asynchronously."""

        cursor = self.responses_collection.find().sort("timestamp", -1)
        documents = await cursor.to_list(length=None)

        fieldnames = [
            "created_at",
            "prompt_id",
            "user_id",
            "purpose",
            "latency_ms",
            "model_info",
        ]
        
        def write_csv():
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
            except IOError as e:
                raise IOError(f"Failed to write CSV file to {output_path}: {str(e)}")
        
        await asyncio.to_thread(write_csv)

    @log_service_call
    async def export_prompt_usage_logs(
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
        
        await self._export_to_csv(output_path)
        return output_path
    
    async def close(self):
        """Close MongoDB connection."""
        self.client.close()
