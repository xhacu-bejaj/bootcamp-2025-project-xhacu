from abc import ABC, abstractmethod
from uuid import uuid4
from typing import Dict, List, Optional, Tuple, TypeAlias

from ..models.domain import Prompt
from app.core.logging import setup_logging, log_api_call



setup_logging()


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
            name: str
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
    @log_api_call
    def create(self, purpose: Purpose, name: str, template: str) -> Prompt:
        new_prompt = Prompt(str(uuid4()), purpose, name, template)
        self._prompts.append(new_prompt)                                            
        return new_prompt
    
    @log_api_call
    def list(self, purpose: Purpose | None) -> list[Prompt]: 
        return [p for p in self._prompts if p.purpose == purpose]
    
    @log_api_call
    def get(self, prompt_id: PromptId) -> Prompt | None:
        return next((p for p in self._prompts if p.id == prompt_id), None)
    
    @log_api_call
    def patch(self, prompt_id: PromptId, name: Optional[str], template: Optional[str]) -> Prompt | None:
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
    def set_active(self, user_id: UserId, purpose: Purpose, prompt_id: PromptId) -> Prompt | None:
        new_active_prompt = next((p for p in self._prompts if p.id == prompt_id), None)
    
        if new_active_prompt is None: # raise exception and log
            return None
        
        active_key = (user_id, purpose)
        
        if active_key in self._active_prompts:
            return None
        
        self._active_prompts[active_key] = prompt_id 
        
        new_active_prompt.active = True
            
        return new_active_prompt

    @log_api_call    
    def get_active(self, user_id: UserId, purpose: Purpose) -> Prompt | None:
        
        active_prompt_id = self._active_prompts.get((user_id, purpose))
        # If user_id does not have an active prompt for purpose return 
        if active_prompt_id is None:
            return None
            
        return next((p for p in self._prompts if p.id == active_prompt_id), None)
    
            
        
    

class FileSnapshotStore(InMemoryStore):
    """Wraps InMemoryStore and snapshots to var/data.json on writes."""
    ...