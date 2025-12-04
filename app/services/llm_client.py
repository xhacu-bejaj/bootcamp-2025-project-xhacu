from abc import ABC, abstractmethod

from app.models.domain import Prompt


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self, active_prompt: Prompt, document_text, **params
    ): 
        ...
