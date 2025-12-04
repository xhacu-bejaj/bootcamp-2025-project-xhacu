from typing import Optional, TypeAlias
from uuid import uuid4
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

from ..models.domain import Prompt
from app.core.logging import log_api_call
from app.core.config import global_settings

UserId: TypeAlias = str
PromptId: TypeAlias = str
Purpose: TypeAlias = str


class MongoDBStore:
    """MongoDB-backed implementation of PromptStore interface."""

    def __init__(self, mongodb_uri: Optional[str] = None):
        """Initialize MongoDB connection and setup collections.

        Args:
            mongodb_uri: Connection string for MongoDB. Defaults to global_settings.MONGODB_URI

        Raises:
            ValueError: If MONGODB_URI is not provided and not in settings
            ConnectionFailure: If unable to connect to MongoDB
        """
        uri = mongodb_uri or global_settings.MONGODB_URI
        if not uri:
            raise ValueError("MONGODB_URI must be provided or set in environment")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command("ping")
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB at {uri}: {e}")

        # Use configured DB and collection names when available (defaults match user-provided setup)
        db_name = getattr(global_settings, "MONGODB_DB_NAME", "data")
        prompts_col_name = getattr(
            global_settings, "MONGODB_PROMPTS_COLLECTION", "prompts_db"
        )
        active_col_name = getattr(
            global_settings, "MONGODB_ACTIVE_COLLECTION", "active_prompts"
        )
        self.db = self.client[db_name]
        self.prompts_collection = self.db[prompts_col_name]
        # active_prompts is stored in a separate collection (created if missing)
        self.active_prompts_collection = self.db[active_col_name]

        # Create indexes for performance
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for common queries."""
        # Index for listing prompts by purpose
        try:
            self.prompts_collection.create_index("purpose")
            self.prompts_collection.create_index([("purpose", 1), ("version", -1)])
        except Exception:
            pass

        # Compound index for active prompt lookups
        try:
            self.active_prompts_collection.create_index(
                [("user_id", 1), ("purpose", 1)], unique=True
            )
        except Exception:
            pass

    def _prompt_from_doc(self, doc: dict) -> Prompt:
        """Convert MongoDB document to Prompt domain object."""
        return Prompt(
            id=str(doc["_id"]), 
            purpose=doc["purpose"],
            name=doc.get("name", "Unnamed"),  
            template=doc.get("template", ""), 
            version=doc.get("version", 1), 
            active=doc.get("active", False),
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
        }

    @log_api_call
    def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
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
        )

        doc = self._prompt_to_doc(prompt)
        self.prompts_collection.insert_one(doc)
        return prompt

    @log_api_call
    def list(self, purpose: Purpose) -> list[Prompt]:
        """List all prompts for a given purpose.

        Args:
            purpose: Filter prompts by this purpose

        Returns:
            List of Prompt objects
        """
        docs = self.prompts_collection.find({"purpose": purpose})
        return [self._prompt_from_doc(doc) for doc in docs]

    @log_api_call
    def get(self, prompt_id: PromptId) -> Prompt | None:
        """Retrieve a prompt by ID.

        Args:
            prompt_id: ID of the prompt to retrieve

        Returns:
            Prompt object or None if not found
        """
        doc = self.prompts_collection.find_one({"_id": prompt_id})
        return self._prompt_from_doc(doc) if doc else None

    @log_api_call
    def patch(
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
            result = self.prompts_collection.find_one_and_update(
                {"_id": prompt_id},
                {"$set": update_dict, "$inc": {"version": 1}},
                return_document=True,
            )

            return self._prompt_from_doc(result) if result else None

        return self.get(prompt_id)

    @log_api_call
    def set_active(
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
        prompt = self.get(prompt_id)
        if not prompt:
            return None

        active_doc = self.active_prompts_collection.find_one(
            {"user_id": user_id, "purpose": purpose}
        )

        if active_doc:
            old_prompt_id = active_doc["prompt_id"]
            self.prompts_collection.update_one(
                {"_id": old_prompt_id}, {"$set": {"active": False}}
            )

        self.active_prompts_collection.update_one(
            {"user_id": user_id, "purpose": purpose},
            {"$set": {"prompt_id": prompt_id}},
            upsert=True,
        )

        self.prompts_collection.update_one(
            {"_id": prompt_id}, {"$set": {"active": True}}
        )

        return self.get(prompt_id)

    @log_api_call
    def get_active(self, user_id: UserId, purpose: Purpose) -> Prompt | None:
        """Retrieve the active prompt for a user and purpose.

        Args:
            user_id: User identifier
            purpose: Prompt purpose/category

        Returns:
            Active Prompt object or None if not set
        """
        active_doc = self.active_prompts_collection.find_one(
            {"user_id": user_id, "purpose": purpose}
        )

        if not active_doc:
            return None

        return self.get(active_doc["prompt_id"])

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
