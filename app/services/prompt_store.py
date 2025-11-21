from abc import ABC, abstractmethod
from uuid import uuid4, UUID
from typing import Dict, List, Tuple, TypeAlias
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
        self._prompts: List[Prompt] = []
        # Contains only prompt_id because it should only do one thing: store a prompt as active for a user, nothing else
        self._active_prompts:Dict[Tuple[UserId,Purpose], PromptId] = {} # Dict of associating user_id with its active_prompts

    # Add checks, purpose, name, template cannot be None otherwise creation must fail
    def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
        new_prompt = Prompt(str(uuid4()), purpose, name, template)
        self._prompts.append(new_prompt)                                            
        return new_prompt
    
    def list(self, purpose: Purpose | None = None) -> List[Prompt]:
        return [p for p in self._prompts if p.purpose == purpose]
    
    def get(self, prompt_id: PromptId) -> Prompt | None:
        return next((p for p in self._prompts if p.id == prompt_id), None)
    
    def patch(self, prompt_id: PromptId, template: str) -> Prompt | None:
        # The prompt_id could be non-existant so add a check for that
        for prompt in self._prompts:
            if prompt.id == prompt_id:
                prompt.template = template
        return prompt
    
    
    def set_active(self, user_id: UserId, purpose: Purpose, prompt_id: PromptId) -> Prompt | None:
        # it must be in self._prompts
        self._active_prompts[(user_id,purpose)] = prompt_id # Add it to _active_prompts ==> prompt_id is active
        for prompt in self._prompts: 
            if prompt.id == prompt_id:
                return prompt
        return None
    
    #self._active_prompts:Dict[Tuple[UserId,Purpose], PromptId]
    # Get THE active prompts for a specific user and a specific purpose
    def get_active(self, user_id: str, purpose: Purpose) -> Prompt | None:
        active_prompt_id = self._active_prompts[(user_id, purpose)]
        # This method signature suggests that a active prompt is identified by user_id and purpose
        for prompt in self._prompts: 
            if prompt.id == active_prompt_id:
                return prompt
        return None
    
    ############# Create a method to easily retrieve from self._prompt and self._active_prompt
            
        
    
    


class FileSnapshotStore(InMemoryStore):
    """Wraps InMemoryStore and snapshots to var/data.json on writes."""
    ...