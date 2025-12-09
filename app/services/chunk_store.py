"""
Chunk storage service using ChromaDB for vector similarity search.

This module provides the ChunkStore class for inserting and retrieving
text chunks from a vector database.
"""

import uuid
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.domain import Chunk
from app.core.logging import log_api_call


class ChunkStore:
    """Service for storing and retrieving text chunks using ChromaDB."""

    def __init__(self):
        """Initialize the ChunkStore with ChromaDB client and collection."""
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  
        )

    @log_api_call
    def insert_chunk(self, text: str, metadata: dict | None = None) -> Chunk:
        """
        Insert a text chunk into the vector database.

        Args:
            text: The text content to store
            metadata: Optional metadata dictionary

        Returns:
            Chunk object with generated ID

        Raises:
            ValueError: If text exceeds maximum chunk length
            RuntimeError: If database insertion fails
        """

        if len(text) > settings.MAX_CHUNK_LENGTH:
            raise ValueError(
                f"Chunk exceeds maximum length of {settings.MAX_CHUNK_LENGTH} characters. "
                f"Got {len(text)} characters."
            )

        chunk_id = str(uuid.uuid4())

        chunk_metadata = metadata or {}
        chunk_metadata["length"] = len(text)

        try:
            self._collection.add(
                ids=[chunk_id],
                documents=[text],
                metadatas=[chunk_metadata]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to insert chunk into database: {str(e)}") from e

        return Chunk(id=chunk_id, text=text, metadata=chunk_metadata)

    @log_api_call
    def retrieve_chunks(self, query_text: str, n_chunks: int = 5) -> List[Chunk]:
        """
        Retrieve the most similar chunks to the query text.

        Args:
            query_text: Text to search for
            n_chunks: Number of chunks to retrieve (default: 5)

        Returns:
            List of Chunk objects ordered by similarity (most similar first)

        Raises:
            ValueError: If n_chunks is invalid
            RuntimeError: If database query fails
        """
        if n_chunks < 1:
            raise ValueError("n_chunks must be at least 1")

        if n_chunks > 100:
            raise ValueError("n_chunks cannot exceed 100")

        try:
            count = self._collection.count()
            if count == 0:
                return []
            
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_chunks, count)
            )

            chunks: List[Chunk] = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    
                    meta: dict = {}
                    if results['metadatas'] and results['metadatas'][0] and results['metadatas'][0][i]:
                        meta = dict(results['metadatas'][0][i])
                    
                    chunk = Chunk(
                        id=chunk_id,
                        text=results['documents'][0][i] if results['documents'] and results['documents'][0] else "",
                        metadata=meta
                    )
                    
                    if results['distances'] and results['distances'][0]:
                        chunk.metadata['distance'] = results['distances'][0][i]
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve chunks from database: {str(e)}") from e

    def count(self) -> int:
        """
        Get the total number of chunks in the database.

        Returns:
            Number of chunks
        """
        return self._collection.count()

    def clear(self) -> None:
        """
        Clear all chunks from the database.

        Warning: This operation cannot be undone.
        """
        self._client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

_chunk_store_instance: ChunkStore | None = None


def get_chunk_store() -> ChunkStore:
    """
    Get the singleton ChunkStore instance.

    Returns:
        ChunkStore instance
    """
    global _chunk_store_instance
    if _chunk_store_instance is None:
        _chunk_store_instance = ChunkStore()
    return _chunk_store_instance
