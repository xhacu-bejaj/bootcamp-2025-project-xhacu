from abc import ABC, abstractmethod
from uuid import uuid4
from typing import List, TypeAlias
import json, os

from ..models.domain import Prompt


UserId: TypeAlias = str
PromptId: TypeAlias = str
Purpose: TypeAlias = str

class PromptStore(ABC):

    @abstractmethod
    def create(
            self,
            purpose: Purpose,
            name: str,
            template: str,
        ) -> Prompt:
        ...

    @abstractmethod
    def list(
            self,
            purpose: Purpose | None = None,
        ) -> list[Prompt]:
        ...

    @abstractmethod
    def get(
            self,
            prompt_id: PromptId,
        ) -> Prompt | None:
        ...

    @abstractmethod
    def patch(
            self,
            prompt_id: PromptId,
            template: str,
        ) -> Prompt | None:
        ...

    @abstractmethod
    def set_active(
            self,
            user_id: UserId,
            purpose: Purpose,
            prompt_id: PromptId,
        ) -> Prompt | None:
        ...

    @abstractmethod
    def get_active(
            self,
            user_id: UserId,
            purpose: Purpose,
        ) -> Prompt | None:
        ...


class InMemoryStore(PromptStore):
    def __init__(self):
        # This way the storage is independent of Prompt
        # I can change the Prompt class and it doesn't affect the InMemoryStore
        self._prompts: List[Prompt] 
        self._active_prompts:List[Prompt]

    def create(self, purpose: str, name: str, template: str) -> Prompt:
        return Prompt(str(uuid4()), purpose, name, template)
    
    def list(self, purpose: str | None = None) -> List[Prompt]:
        # Let the option str | None so that it returns all prompts if none is specified
        return [p for p in self._prompts if p.purpose != purpose]
    
    


class FileSnapshotStore(InMemoryStore):
    """Wraps InMemoryStore and snapshots to var/data.json on writes."""
    ...