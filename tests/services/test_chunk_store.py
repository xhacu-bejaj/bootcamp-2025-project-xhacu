"""
Tests for the ChunkStore service.

This module tests the vector database functionality for storing
and retrieving text chunks.
"""

import pytest
from app.services.chunk_store import ChunkStore, get_chunk_store
from app.models.domain import Chunk


@pytest.fixture
def chunk_store():
    """Create a ChunkStore instance and clear it after each test."""
    store = ChunkStore()
    yield store
    # Clean up after test
    store.clear()


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
        text = "x" * 1001  # Exceeds MAX_CHUNK_LENGTH of 1000
        
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
        # All IDs should be unique
        ids = [chunk.id for chunk in inserted_chunks]
        assert len(ids) == len(set(ids))


class TestChunkStoreRetrieve:
    """Tests for retrieving chunks from the vector database."""

    def test_retrieve_similar_chunks(self, chunk_store):
        """Test retrieving semantically similar chunks."""
        # Insert chunks with related content
        chunk_store.insert_chunk("Python is a programming language.")
        chunk_store.insert_chunk("JavaScript is used for web development.")
        chunk_store.insert_chunk("Dogs are popular pets.")
        
        # Query for programming-related content
        results = chunk_store.retrieve_chunks("coding languages", n_chunks=2)
        
        assert len(results) <= 2
        assert all(isinstance(chunk, Chunk) for chunk in results)
        # Should retrieve programming-related chunks
        assert any("Python" in chunk.text or "JavaScript" in chunk.text for chunk in results)

    def test_retrieve_chunks_empty_database(self, chunk_store):
        """Test retrieving from an empty database."""
        results = chunk_store.retrieve_chunks("test query", n_chunks=5)
        
        assert results == []

    def test_retrieve_chunks_limit_validation(self, chunk_store):
        """Test that n_chunks validation works."""
        chunk_store.insert_chunk("Test chunk")
        
        # Test minimum limit
        with pytest.raises(ValueError, match="must be at least 1"):
            chunk_store.retrieve_chunks("query", n_chunks=0)
        
        # Test maximum limit
        with pytest.raises(ValueError, match="cannot exceed 100"):
            chunk_store.retrieve_chunks("query", n_chunks=101)

    def test_retrieve_chunks_respects_limit(self, chunk_store):
        """Test that retrieve respects the n_chunks limit."""
        # Insert 5 chunks
        for i in range(5):
            chunk_store.insert_chunk(f"Chunk number {i}")
        
        # Request only 2 chunks
        results = chunk_store.retrieve_chunks("chunk", n_chunks=2)
        
        assert len(results) == 2

    def test_retrieve_chunks_includes_metadata(self, chunk_store):
        """Test that retrieved chunks include metadata."""
        metadata = {"source": "test.txt", "position": 42}
        chunk_store.insert_chunk("Test chunk with metadata", metadata=metadata)
        
        results = chunk_store.retrieve_chunks("test chunk", n_chunks=1)
        
        assert len(results) == 1
        assert results[0].metadata["source"] == "test.txt"
        assert results[0].metadata["position"] == 42
        assert "length" in results[0].metadata
        assert "distance" in results[0].metadata

    def test_semantic_search_finds_related_content(self, chunk_store):
        """Test that semantic search finds semantically similar content."""
        # Insert chunks about different topics
        chunk_store.insert_chunk("Python is a programming language for data science.")
        chunk_store.insert_chunk("Machine learning algorithms can identify patterns.")
        chunk_store.insert_chunk("Dogs are loyal pets and companions.")
        chunk_store.insert_chunk("Pizza is an Italian dish with cheese.")
        
        # Search for AI/coding - should find programming and ML chunks
        results = chunk_store.retrieve_chunks("artificial intelligence and coding", n_chunks=2)
        
        assert len(results) == 2
        # The top results should be about programming/ML, not dogs or pizza
        result_texts = [chunk.text for chunk in results]
        assert any("Python" in text or "Machine learning" in text for text in result_texts)
        assert not any("Dogs" in text or "Pizza" in text for text in result_texts)

    def test_semantic_search_ranks_by_similarity(self, chunk_store):
        """Test that results are ordered by semantic similarity."""
        # Insert chunks with varying relevance to query
        chunk_store.insert_chunk("Neural networks are used in deep learning.")
        chunk_store.insert_chunk("The weather is sunny today.")
        chunk_store.insert_chunk("Artificial intelligence transforms technology.")
        
        results = chunk_store.retrieve_chunks("machine learning and AI", n_chunks=3)
        
        assert len(results) == 3
        # First result should have the smallest distance (most similar)
        distances = [chunk.metadata["distance"] for chunk in results]
        assert distances == sorted(distances)  # Should be in ascending order
        
        # Most relevant chunks should be about AI/ML
        assert "intelligence" in results[0].text.lower() or "neural" in results[0].text.lower()
        # Least relevant should be about weather
        assert "weather" in results[-1].text.lower()


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
        # Insert some chunks
        chunk_store.insert_chunk("Chunk 1")
        chunk_store.insert_chunk("Chunk 2")
        assert chunk_store.count() == 2
        
        # Clear the database
        chunk_store.clear()
        assert chunk_store.count() == 0

    def test_get_chunk_store_singleton(self):
        """Test that get_chunk_store returns a singleton instance."""
        store1 = get_chunk_store()
        store2 = get_chunk_store()
        
        assert store1 is store2
