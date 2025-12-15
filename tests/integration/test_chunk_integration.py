"""
Integration tests for ChunkStore with a real ChromaDB instance.
"""

import pytest
import asyncio
from app.services.chunk_store import ChunkStore

# The `real_chunk_store` fixture is defined in tests/integration/conftest.py
# and provides a clean ChromaDB instance for each test.

@pytest.mark.integration
class TestChunkStoreIntegration:
    """Integration tests for ChromaDB chunk operations."""

    def test_insert_and_count_chunks(self, real_chunk_store: ChunkStore):
        """Test inserting chunks and counting them."""
        store = real_chunk_store
        
        chunk1 = store.insert_chunk(text="First test chunk.", metadata={"source": "doc1"})
        chunk2 = store.insert_chunk(text="Second test chunk.", metadata={"source": "doc1"})
        
        assert chunk1.id is not None
        assert chunk2.id is not None
        assert store.count() == 2

    def test_semantic_search_relevance(self, real_chunk_store: ChunkStore):
        """Test that semantic search returns relevant results."""
        store = real_chunk_store
        
        store.insert_chunk(text="Python is a programming language.", metadata={"topic": "programming"})
        store.insert_chunk(text="The python snake is a constrictor.", metadata={"topic": "animals"})
        store.insert_chunk(text="JavaScript is used for web development.", metadata={"topic": "programming"})
        
        results = store.retrieve_chunks("software development", n_chunks=2)
        
        assert len(results) == 2
        # The programming chunks should be more relevant than the animal one.
        topics = {r.metadata.get("topic") for r in results}
        assert "programming" in topics
        assert "animals" not in topics

    def test_distance_scores_are_present(self, real_chunk_store: ChunkStore):
        """Test that distance scores are included and sorted."""
        store = real_chunk_store
        
        store.insert_chunk(text="Machine learning is a subset of AI.")
        store.insert_chunk(text="Baking a cake requires flour and sugar.")
        
        results = store.retrieve_chunks("artificial intelligence", n_chunks=2)
        
        assert len(results) == 2
        assert "distance" in results[0].metadata
        assert "distance" in results[1].metadata
        # The first result should be more similar (smaller distance) than the second.
        assert results[0].metadata["distance"] < results[1].metadata["distance"]
        assert "Machine learning" in results[0].text

    def test_retrieve_with_metadata(self, real_chunk_store: ChunkStore):
        """Test retrieving chunks preserves metadata."""
        store = real_chunk_store
        store.insert_chunk(text="Chunk from report A.", metadata={"source": "reportA.pdf", "page": 1})
        
        results = store.retrieve_chunks("report A", n_chunks=1)
        
        assert len(results) == 1
        retrieved_chunk = results[0]
        assert "source" in retrieved_chunk.metadata
        assert retrieved_chunk.metadata["source"] == "reportA.pdf"
        assert retrieved_chunk.metadata["page"] == 1

    def test_clear_collection(self, real_chunk_store: ChunkStore):
        """Test clearing all chunks from the collection."""
        store = real_chunk_store
        
        store.insert_chunk(text="Chunk 1")
        assert store.count() == 1
        
        store.clear()
        assert store.count() == 0
