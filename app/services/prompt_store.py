from abc import ABC, abstractmethod
from uuid import uuid4
from typing import Dict, List, Tuple, TypeAlias
import json
import os
import asyncio

from ..models.domain import Prompt
from app.core.logging import setup_logging, log_api_call

setup_logging()

UserId: TypeAlias = str
PromptId: TypeAlias = str
Purpose: TypeAlias = str


class PromptStore(ABC):
    @abstractmethod
    async def create(
        self,
        purpose: Purpose,
        name: str,
        template: str,
    ) -> Prompt: ...

    @abstractmethod
    async def list(
        self,
        purpose: Purpose | None = None, # maybe add filtering by user_id later
    ) -> list[Prompt]: ...

    @abstractmethod
    async def get(
        self,
        prompt_id: PromptId,
    ) -> Prompt | None: ...

    @abstractmethod
    async def patch(self, prompt_id: PromptId, template: str, name: str) -> Prompt | None: ...

    @abstractmethod
    async def set_active(
        self,
        user_id: UserId,
        purpose: Purpose,
        prompt_id: PromptId,
    ) -> Prompt | None: ...

    @abstractmethod
    async def get_active(
        self,
        user_id: UserId,
        purpose: Purpose,
    ) -> Prompt | None: ...


class InMemoryStore(PromptStore):
    def __init__(self):
        self._prompts: List[Prompt] = []
        # Dict of associating user_id with its active_prompts
        self._active_prompts: Dict[Tuple[UserId, Purpose], PromptId] = {}

    @log_api_call
    async def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
        new_prompt = Prompt(str(uuid4()), purpose, name, template)
        self._prompts.append(new_prompt)
        return new_prompt

    @log_api_call
    async def list(self, purpose: Purpose) -> list[Prompt]:  # | None
        return [p for p in self._prompts if p.purpose == purpose]

    @log_api_call
    async def get(self, prompt_id: PromptId) -> Prompt | None:
        return next((p for p in self._prompts if p.id == prompt_id), None)

    @log_api_call
    async def patch(
        self, prompt_id: PromptId, name: str | None = None, template: str | None = None
    ) -> Prompt | None:
        for prompt in self._prompts:
            if prompt.id == prompt_id:
                is_updated = False

                if name is not None:
                    prompt.name = name
                    is_updated = True

                if template is not None:
                    prompt.template = template
                    is_updated = True

                if is_updated:
                    prompt.version += 1

                return prompt

        return None

    @log_api_call
    async def set_active(
        self, user_id: UserId, purpose: Purpose, prompt_id: PromptId
    ) -> Prompt | None:
        new_active_prompt = next((p for p in self._prompts if p.id == prompt_id), None)
        if new_active_prompt is None:
            return None

        active_key = (user_id, purpose)
        old_prompt_id = self._active_prompts.get(active_key)
        if old_prompt_id:
            old_prompt = next((p for p in self._prompts if p.id == old_prompt_id), None)
            if old_prompt:
                old_prompt.active = False

        self._active_prompts[active_key] = prompt_id
        new_active_prompt.active = True

        return new_active_prompt

    @log_api_call
    async def get_active(self, user_id: UserId, purpose: Purpose) -> Prompt | None:
        active_prompt_id = self._active_prompts.get((user_id, purpose))
    
        if active_prompt_id is None:
            return None

        return next((p for p in self._prompts if p.id == active_prompt_id), None)


class FileSnapshotStore(InMemoryStore):
    """Wraps InMemoryStore and snapshots to var/data.json on writes."""

    def __init__(self, filepath: str = "var/data.json"):
        """Initialize FileSnapshotStore with optional filepath.

        Args:
            filepath: Path to JSON file for persistence (default: var/data.json)
        """
        super().__init__()
        self.filepath = filepath
        self._load_sync()

    def _load_sync(self):
        """Load data from JSON file if it exists.
        
        Deserializes prompts and active_prompts from JSON file.
        If file doesn't exist or is corrupted, initializes empty state.
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._prompts = [
                    Prompt(
                        id=p["id"],
                        purpose=p["purpose"],
                        name=p["name"],
                        template=p["template"],
                        version=p.get("version", 1),
                        active=p.get("active", False),
                    )
                    for p in data.get("prompts", [])
                ]

                self._active_prompts = {
                    (entry["user_id"], entry["purpose"]): entry["prompt_id"]
                    for entry in data.get("active_prompts", [])
                }
            except (json.JSONDecodeError, KeyError, IOError, OSError) as e:
                # If file is corrupted or unreadable, start with empty state
                print(f"Warning: Could not load {self.filepath}: {e}. Starting with empty state.")
                self._prompts = []
                self._active_prompts = {}

    def _save_sync(self):
        """Save current state to JSON file.
        
        Serializes all prompts and active_prompts to JSON file.
        Creates parent directories if they don't exist.
        
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create parent directory if filepath includes a directory
            dir_path = os.path.dirname(self.filepath)
            if dir_path:  # Only create if there's a directory component
                os.makedirs(dir_path, exist_ok=True)

            data = {
                "prompts": [
                    {
                        "id": p.id,
                        "purpose": p.purpose,
                        "name": p.name,
                        "template": p.template,
                        "version": p.version,
                        "active": p.active,
                    }
                    for p in self._prompts
                ],
                "active_prompts": [
                    {
                        "user_id": user_id,
                        "purpose": purpose,
                        "prompt_id": prompt_id,
                    }
                    for (user_id, purpose), prompt_id in self._active_prompts.items()
                ],
            }

            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            print(f"Error: Failed to save to {self.filepath}: {e}")
            raise IOError(f"Failed to save prompts to {self.filepath}: {e}")

    async def _save(self):
        await asyncio.to_thread(self._save_sync)

    async def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
        """Create a new prompt and save to file.
        
        Args:
            purpose: Category/purpose of the prompt
            name: Human-readable name
            template: Jinja2 template string
            
        Returns:
            Newly created Prompt instance
        """
        result = await super().create(purpose, name, template)
        await self._save()
        return result

    async def patch(
        self, prompt_id: PromptId, name: str | None = None, template: str | None = None
    ) -> Prompt | None:
        """Update a prompt and save to file.
        
        Args:
            prompt_id: ID of prompt to update
            name: New name (optional)
            template: New template (optional)
            
        Returns:
            Updated Prompt instance or None if not found
        """
        result = await super().patch(prompt_id, name, template)
        if result is not None:
            await self._save()
        return result

    async def set_active(
        self, user_id: UserId, purpose: Purpose, prompt_id: PromptId
    ) -> Prompt | None:
        """Set active prompt and save to file.
        
        Args:
            user_id: User identifier
            purpose: Prompt purpose/category
            prompt_id: ID of prompt to set as active
            
        Returns:
            Activated Prompt instance or None if not found
        """
        result = await super().set_active(user_id, purpose, prompt_id)
        if result is not None:
            await self._save()
        return result
