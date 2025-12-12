from abc import ABC, abstractmethod

from app.models.schemas import LLMOutput


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **params) -> LLMOutput | None: #explain why LLMOutput or None
        ...
