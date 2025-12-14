from abc import ABC, abstractmethod

from app.models.domain import Chunk

class VectorStore(ABC):
    @abstractmethod
    def insert_chunk(self, text: str, metadata: dict | None = None) -> Chunk:
        ...

    @abstractmethod
    def retrieve_chunks(self, query_text: str, n_chunks: int = 5) -> list[Chunk]:
        ...