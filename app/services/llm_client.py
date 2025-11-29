from abc import ABC, abstractmethod

from app.models.domain import Prompt


class LLMClient(ABC):
    @abstractmethod
    def generate(self, template: str| None, document_text, **params): #-> PredictResponse | None: 
        ...



