"""
Tests for the ChunkStore service.

This module tests the vector database functionality for storing
and retrieving text chunks.
"""

import pytest
from app.services.chunk_store import ChunkStore
from app.models.domain import Chunk
from app.core.config import settings  # Import settings

# This is a synchronous fixture now
@pytest.fixture
def chunk_store():
    """Create a ChunkStore instance and clear it after each test."""
    store = ChunkStore()
    store.clear()  # Clear at the beginning
    yield store
    store.clear()  # Clean up after test

class TestChunkStoreInsert:
    """Tests for inserting chunks into the vector database."""

    def test_insert_chunk_success(self, chunk_store):
        """Test that a chunk can be successfully inserted."""
        text = "This is a test chunk for insertion."
        metadata = {"source": "test_document", "position": 0}
        
        chunk = chunk_store.insert_chunk(text=text, metadata=metadata)
        
        assert chunk.id is not None
        assert chunk.text == text
        assert chunk.metadata["length"] == len(text)
        assert chunk.metadata["source"] == "test_document"
        assert chunk.metadata["position"] == 0

    def test_insert_chunk_without_metadata(self, chunk_store):
        """Test inserting a chunk without metadata."""
        text = "Simple chunk without metadata."
        
        chunk = chunk_store.insert_chunk(text=text)
        
        assert chunk.id is not None
        assert chunk.text == text
        assert chunk.metadata["length"] == len(text)

    def test_insert_chunk_exceeds_max_length(self, chunk_store):
        """Test that inserting a chunk exceeding max length raises ValueError."""
        text = "x" * (settings.MAX_CHUNK_LENGTH + 1)
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            chunk_store.insert_chunk(text=text)

    def test_insert_multiple_chunks(self, chunk_store):
        """Test inserting multiple chunks."""
        chunks_data = [
            "First chunk about Python programming.",
            "Second chunk about machine learning.",
            "Third chunk about data science."
        ]
        
        inserted_chunks = []
        for text in chunks_data:
            chunk = chunk_store.insert_chunk(text=text)
            inserted_chunks.append(chunk)
        
        assert len(inserted_chunks) == 3
        assert all(chunk.id is not None for chunk in inserted_chunks)
        ids = [chunk.id for chunk in inserted_chunks]
        assert len(ids) == len(set(ids))

class TestChunkStoreBatchInsert:
    """Tests for batch inserting chunks."""

    def test_insert_chunks_success(self, chunk_store):
        """Test successful batch insertion of multiple chunks."""
        texts = ["Batch chunk 1", "Batch chunk 2", "Batch chunk 3"]
        metadatas = [{"source": "batch_test"}, {"source": "batch_test"}, {"source": "batch_test"}]
        
        initial_count = chunk_store.count()
        
        chunks = chunk_store.insert_chunks(texts=texts, metadatas=metadatas)
        
        assert len(chunks) == 3
        assert chunk_store.count() == initial_count + 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].metadata["source"] == "batch_test"

    def test_insert_chunks_mismatched_length_raises_error(self, chunk_store):
        """Test that mismatched texts and metadatas lists raise ValueError."""
        texts = ["Batch chunk 1", "Batch chunk 2"]
        metadatas = [{"source": "batch_test"}]  # Only one metadata
        
        with pytest.raises(ValueError, match="number of texts and metadatas must be the same"):
            chunk_store.insert_chunks(texts=texts, metadatas=metadatas)

    def test_insert_chunks_one_exceeds_max_length_raises_error(self, chunk_store):
        """Test batch insertion fails if any chunk exceeds max length."""
        texts = ["This one is fine.", "x" * (settings.MAX_CHUNK_LENGTH + 1)]
        metadatas = [{}, {}]
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            chunk_store.insert_chunks(texts=texts, metadatas=metadatas)
        
        # Ensure no chunks were added
        assert chunk_store.count() == 0

class TestChunkStoreRetrieve:
    """Tests for retrieving chunks from the vector database."""

    def test_retrieve_similar_chunks(self, chunk_store):
        """Test retrieving semantically similar chunks."""
        chunk_store.insert_chunk("Python is a programming language.")
        chunk_store.insert_chunk("JavaScript is used for web development.")
        chunk_store.insert_chunk("Dogs are popular pets.")
        
        results = chunk_store.retrieve_chunks("coding languages", n_chunks=2)
        
        assert len(results) <= 2
        assert all(isinstance(chunk, Chunk) for chunk in results)
        assert any("Python" in chunk.text or "JavaScript" in chunk.text for chunk in results)

    def test_retrieve_chunks_empty_database(self, chunk_store):
        """Test retrieving from an empty database."""
        results = chunk_store.retrieve_chunks("test query", n_chunks=5)
        assert results == []

    def test_retrieve_chunks_limit_validation(self, chunk_store):
        """Test that n_chunks validation works."""
        chunk_store.insert_chunk("Test chunk")
        
        with pytest.raises(ValueError, match="must be at least 1"):
            chunk_store.retrieve_chunks("query", n_chunks=0)
        
        with pytest.raises(ValueError, match="cannot exceed 100"):
            chunk_store.retrieve_chunks("query", n_chunks=101)

class TestChunkStoreUtility:
    """Tests for utility methods of ChunkStore."""

    def test_count_chunks(self, chunk_store):
        """Test counting the number of chunks in the database."""
        assert chunk_store.count() == 0
        
        chunk_store.insert_chunk("First chunk")
        assert chunk_store.count() == 1
        
        chunk_store.insert_chunk("Second chunk")
        assert chunk_store.count() == 2

    def test_clear_database(self, chunk_store):
        """Test clearing all chunks from the database."""
        chunk_store.insert_chunk("Chunk 1")
        chunk_store.insert_chunk("Chunk 2")
        assert chunk_store.count() == 2
        
        chunk_store.clear()
        assert chunk_store.count() == 0
