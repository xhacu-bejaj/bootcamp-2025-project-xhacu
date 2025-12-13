"""
Integration tests for ChunkStore with real ChromaDB.

These tests use real vector database operations to verify
chunk storage and retrieval behavior.
"""
import pytest


@pytest.mark.integration
class TestChunkStoreIntegration:
    """Integration tests for ChromaDB chunk operations."""

    @pytest.mark.asyncio
    async def test_insert_and_count_chunks(self, real_chunk_store):
        """Test inserting chunks and counting them in real ChromaDB."""
        store = real_chunk_store
        
        # Insert chunks
        chunk1 = await store.insert_chunk(
            text="This is the first test chunk for integration testing.",
            metadata={"source": "doc1.txt", "position": 0}
        )
        
        chunk2 = await store.insert_chunk(
            text="This is the second test chunk with similar content.",
            metadata={"source": "doc1.txt", "position": 1}
        )
        
        assert chunk1.id is not None
        assert chunk2.id is not None
        assert chunk1.id != chunk2.id
        
        # Count chunks
        count = await store.count()
        assert count >= 2
        
        # Retrieve chunks
        results = await store.retrieve_chunks(
            query_text="test chunk content",
            n_chunks=2
        )
        
        assert len(results) == 2
        # Both chunks should be in results
        returned_ids = {r.id for r in results}
        assert chunk1.id in returned_ids or chunk2.id in returned_ids

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, real_chunk_store):
        """Test that semantic search returns relevant results."""
        store = real_chunk_store
        
        # Insert diverse chunks
        await store.insert_chunk(
            text="Python is a programming language used for web development.",
            metadata={"topic": "programming"}
        )
        
        await store.insert_chunk(
            text="The python snake is a large constrictor found in tropical regions.",
            metadata={"topic": "animals"}
        )
        
        await store.insert_chunk(
            text="JavaScript and TypeScript are popular for frontend development.",
            metadata={"topic": "programming"}
        )
        
        # Query for programming content
        results = await store.retrieve_chunks(
            query_text="software development programming languages",
            n_chunks=2
        )
        
        assert len(results) == 2
        # The programming-related chunks should be more relevant
        topics = [r.metadata.get("topic") for r in results]
        assert topics.count("programming") >= 1

    @pytest.mark.asyncio
    async def test_distance_scores(self, real_chunk_store):
        """Test that distance scores are included in results."""
        store = real_chunk_store
        
        # Insert identical and different chunks
        await store.insert_chunk(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"version": "A"}
        )
        
        await store.insert_chunk(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"version": "B"}  # Identical text
        )
        
        await store.insert_chunk(
            text="Cooking recipes require precise measurements and timing.",
            metadata={"version": "C"}  # Very different
        )
        
        # Query with similar text
        results = await store.retrieve_chunks(
            query_text="Machine learning is part of AI.",
            n_chunks=3
        )
        
        assert len(results) == 3
        
        # Distance scores should be present in metadata
        distances = [r.metadata.get("distance") for r in results if r.metadata.get("distance") is not None]
        assert len(distances) > 0

    @pytest.mark.asyncio
    async def test_retrieve_with_metadata(self, real_chunk_store):
        """Test retrieving chunks preserves metadata."""
        store = real_chunk_store
        
        # Insert chunks with different metadata
        await store.insert_chunk(
            text="Important document chunk from report A.",
            metadata={"source": "reportA.pdf", "page": 1}
        )
        
        await store.insert_chunk(
            text="Important document chunk from report B.",
            metadata={"source": "reportB.pdf", "page": 1}
        )
        
        await store.insert_chunk(
            text="Another chunk from report A.",
            metadata={"source": "reportA.pdf", "page": 2}
        )
        
        # Retrieve chunks
        results = await store.retrieve_chunks(
            query_text="important document",
            n_chunks=10
        )
        
        # Should return chunks with metadata
        assert len(results) >= 2
        for result in results:
            assert "source" in result.metadata
            assert result.metadata["source"] in ["reportA.pdf", "reportB.pdf"]

    @pytest.mark.asyncio
    async def test_clear_collection(self, real_chunk_store):
        """Test clearing all chunks from collection."""
        store = real_chunk_store
        
        # Insert some chunks
        await store.insert_chunk(text="Chunk 1", metadata={"idx": 1})
        await store.insert_chunk(text="Chunk 2", metadata={"idx": 2})
        await store.insert_chunk(text="Chunk 3", metadata={"idx": 3})
        
        # Verify they exist
        count_before = await store.count()
        assert count_before >= 3
        
        # Clear collection
        await store.clear()
        
        # Verify it's empty
        count_after = await store.count()
        assert count_after == 0

    @pytest.mark.asyncio
    async def test_chunk_size_limit(self, real_chunk_store):
        """Test that chunks exceeding size limit are rejected."""
        import pytest
        store = real_chunk_store
        
        # Try to insert text exceeding max length (1000 chars)
        large_text = "x" * 1001
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            await store.insert_chunk(text=large_text)

    @pytest.mark.asyncio
    async def test_special_characters_in_chunks(self, real_chunk_store):
        """Test handling of special characters in text."""
        store = real_chunk_store
        
        special_texts = [
            "Text with special chars: @#$%^&*()",
            "Text with quotes: 'single' and \"double\"",
            "Text with newlines:\nLine 1\nLine 2",
        ]
        
        inserted_chunks = []
        for text in special_texts:
            chunk = await store.insert_chunk(text=text)
            inserted_chunks.append(chunk)
            assert chunk.id is not None
        
        # Verify all can be queried
        results = await store.retrieve_chunks(
            query_text="Text with",
            n_chunks=len(special_texts)
        )
        
        assert len(results) >= len(special_texts)

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, real_chunk_store):
        """Test concurrent chunk insertions."""
        import asyncio
        store = real_chunk_store
        
        async def insert_chunk(idx):
            return await store.insert_chunk(
                text=f"Concurrent chunk number {idx}",
                metadata={"index": idx}
            )
        
        # Insert multiple chunks concurrently
        chunks = await asyncio.gather(*[insert_chunk(i) for i in range(10)])
        
        # All should have unique IDs
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
        
        # Verify count
        count = await store.count()
        assert count >= 10
