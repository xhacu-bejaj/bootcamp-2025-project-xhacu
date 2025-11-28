from abc import ABC, abstractmethod
from ..models.schemas import PredictResponse

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **params) -> PredictResponse: ...




