"""
Chunk storage service using ChromaDB for vector similarity search.

This module provides the ChunkStore class for inserting and retrieving
text chunks from a vector database.
"""

import uuid
from typing import List

from app.core.config import settings
from app.models.domain import Chunk
from app.core.logging import log_api_call
from app.services.vector_store import VectorStore


class ChunkStore(VectorStore):
    """Service for storing and retrieving text chunks using ChromaDB."""

    def __init__(self):
        """Initialize the ChunkStore with ChromaDB client."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} 
        )

    def initialize(self):
        """Asynchronously get or create the collection (no-op since initialized in __init__)."""
        pass

    
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

    def insert_chunks(self, texts: List[str], metadatas: List[dict]) -> List[Chunk]:
        """
        Insert multiple text chunks into the vector database in a single batch.

        Args:
            texts: A list of text content to store.
            metadatas: A list of metadata dictionaries, one for each text.

        Returns:
            A list of Chunk objects with their generated IDs.

        Raises:
            ValueError: If the number of texts and metadatas don't match, or if any
                        text exceeds the maximum chunk length.
            RuntimeError: If the database insertion fails.
        """
        if len(texts) != len(metadatas):
            raise ValueError("The number of texts and metadatas must be the same.")

        chunk_ids = [str(uuid.uuid4()) for _ in texts]
        processed_metadatas = []
        for i, text in enumerate(texts):
            if len(text) > settings.MAX_CHUNK_LENGTH:
                raise ValueError(
                    f"Chunk at index {i} exceeds maximum length of "
                    f"{settings.MAX_CHUNK_LENGTH} characters. Got {len(text)} characters."
                )
            meta = metadatas[i] or {}
            meta["length"] = len(text)
            processed_metadatas.append(meta)

        try:
            self._collection.add(
                ids=chunk_ids,
                documents=texts,
                metadatas=processed_metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to insert chunks into database: {str(e)}") from e

        return [Chunk(id=chunk_ids[i], text=texts[i], metadata=processed_metadatas[i]) for i in range(len(texts))]

    
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
        if n_chunks < 1: # magic number no good
            raise ValueError("n_chunks must be at least 1")

        if n_chunks > 100: # magic number no good
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
