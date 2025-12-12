"""
Tests for the chunks API endpoints.

This module tests the REST API for chunk insertion and retrieval.
"""

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def clear_chunk_store(client):
    """Clear the chunk store before and after each test."""
    store = client.app.state.chunk_store
    await store.clear()
    yield
    await store.clear()


class TestChunkInsertAPI:
    """Tests for POST /v1/chunks/insert endpoint."""

    @pytest.mark.asyncio
    async def test_insert_chunk_success(self, client, clear_chunk_store):
        """Test successfully inserting a chunk."""
        payload = {
            "text": "This is a test chunk for the API.",
            "metadata": {
                "source": "test_document.txt",
                "position": 0
            }
        }
        
        response = client.post("/v1/chunks/insert", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["text"] == payload["text"]
        assert data["metadata"]["length"] == len(payload["text"])
        assert data["metadata"]["source"] == "test_document.txt"
        assert data["metadata"]["position"] == 0
        assert data["distance"] is None

    @pytest.mark.asyncio
    async def test_insert_chunk_without_metadata(self, client, clear_chunk_store):
        """Test inserting a chunk without metadata."""
        payload = {
            "text": "Simple chunk without metadata."
        }
        
        response = client.post("/v1/chunks/insert", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["text"] == payload["text"]
        assert data["metadata"]["length"] == len(payload["text"])
        assert data["metadata"]["source"] is None
        assert data["metadata"]["position"] is None

    @pytest.mark.asyncio
    async def test_insert_chunk_exceeds_max_length(self, client, clear_chunk_store):
        """Test that inserting a chunk exceeding max length returns 400."""
        payload = {
            "text": "x" * 1001  # Exceeds MAX_CHUNK_LENGTH of 1000
        }
        
        response = client.post("/v1/chunks/insert", json=payload)
        
        assert response.status_code == 400
        assert "exceeds maximum length" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_insert_chunk_missing_text(self, client, clear_chunk_store):
        """Test that missing text field returns 422."""
        payload = {
            "metadata": {"source": "test"}
        }
        
        response = client.post("/v1/chunks/insert", json=payload)
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_insert_multiple_chunks(self, client, clear_chunk_store):
        """Test inserting multiple chunks."""
        chunks = [
            {"text": "First chunk"},
            {"text": "Second chunk"},
            {"text": "Third chunk"}
        ]
        
        inserted_ids = []
        for chunk in chunks:
            response = client.post("/v1/chunks/insert", json=chunk)
            assert response.status_code == 201
            inserted_ids.append(response.json()["id"])
        
        # All IDs should be unique
        assert len(inserted_ids) == len(set(inserted_ids))


class TestChunkRetrieveAPI:
    """Tests for GET /v1/chunks/retrieve endpoint."""

    @pytest.mark.asyncio
    async def test_retrieve_chunks_success(self, client, clear_chunk_store):
        """Test successfully retrieving chunks."""
        # Insert test chunks
        chunks_data = [
            {"text": "Python is a programming language."},
            {"text": "JavaScript is used for web development."},
            {"text": "Machine learning is a subset of AI."}
        ]
        
        for chunk in chunks_data:
            client.post("/v1/chunks/insert", json=chunk)
        
        # Query for programming-related content
        response = client.get("/v1/chunks/retrieve?text=programming languages&n_chunks=2")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 2
        assert all("id" in chunk for chunk in data)
        assert all("text" in chunk for chunk in data)
        assert all("metadata" in chunk for chunk in data)

    @pytest.mark.asyncio
    async def test_retrieve_chunks_default_limit(self, client, clear_chunk_store):
        """Test that default n_chunks is 5."""
        # Insert 10 chunks
        for i in range(10):
            client.post("/v1/chunks/insert", json={"text": f"Chunk number {i}"})
        
        # Query without specifying n_chunks
        response = client.get("/v1/chunks/retrieve?text=chunk")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5  # Default limit

    @pytest.mark.asyncio
    async def test_retrieve_chunks_custom_limit(self, client, clear_chunk_store):
        """Test retrieving with custom n_chunks."""
        # Insert 5 chunks
        for i in range(5):
            client.post("/v1/chunks/insert", json={"text": f"Test chunk {i}"})
        
        # Request only 2 chunks
        response = client.get("/v1/chunks/retrieve?text=test&n_chunks=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_retrieve_chunks_empty_database(self, client, clear_chunk_store):
        """Test retrieving from empty database."""
        response = client.get("/v1/chunks/retrieve?text=anything&n_chunks=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_retrieve_chunks_missing_text(self, client, clear_chunk_store):
        """Test that missing text parameter returns 422."""
        response = client.get("/v1/chunks/retrieve?n_chunks=5")
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_retrieve_chunks_empty_text(self, client, clear_chunk_store):
        """Test that empty text parameter returns 400."""
        response = client.get("/v1/chunks/retrieve?text=&n_chunks=5")
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_retrieve_chunks_invalid_n_chunks_too_low(self, client, clear_chunk_store):
        """Test that n_chunks < 1 returns 400."""
        client.post("/v1/chunks/insert", json={"text": "Test"})
        
        response = client.get("/v1/chunks/retrieve?text=test&n_chunks=0")
        
        assert response.status_code == 400
        assert "must be between 1 and 100" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_retrieve_chunks_invalid_n_chunks_too_high(self, client, clear_chunk_store):
        """Test that n_chunks > 100 returns 400."""
        client.post("/v1/chunks/insert", json={"text": "Test"})
        
        response = client.get("/v1/chunks/retrieve?text=test&n_chunks=101")
        
        assert response.status_code == 400
        assert "must be between 1 and 100" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_retrieve_chunks_includes_distance(self, client, clear_chunk_store):
        """Test that retrieved chunks include distance information."""
        client.post("/v1/chunks/insert", json={"text": "Sample text for testing"})
        
        response = client.get("/v1/chunks/retrieve?text=sample&n_chunks=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        # Distance can be None or a float value
        assert "distance" in data[0]

    @pytest.mark.asyncio
    async def test_retrieve_chunks_preserves_metadata(self, client, clear_chunk_store):
        """Test that retrieved chunks preserve metadata."""
        payload = {
            "text": "Test chunk with metadata",
            "metadata": {
                "source": "document.pdf",
                "position": 42
            }
        }
        client.post("/v1/chunks/insert", json=payload)
        
        response = client.get("/v1/chunks/retrieve?text=test chunk&n_chunks=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["metadata"]["source"] == "document.pdf"
        assert data[0]["metadata"]["position"] == 42
        assert "length" in data[0]["metadata"]


class TestChunkAPIIntegration:
    """Integration tests for chunk API endpoints."""

    @pytest.mark.asyncio
    async def test_insert_and_retrieve_workflow(self, client, clear_chunk_store):
        """Test complete workflow of inserting and retrieving chunks."""
        documents = [
            {"text": "Python is great for data science and machine learning."},
            {"text": "JavaScript is essential for modern web development."},
            {"text": "SQL is used for database management and queries."}
        ]
        
        inserted_ids = []
        for doc in documents:
            response = client.post("/v1/chunks/insert", json=doc)
            assert response.status_code == 201
            inserted_ids.append(response.json()["id"])
        
        # Step 2: Retrieve programming-related chunks
        response = client.get("/v1/chunks/retrieve?text=programming languages&n_chunks=2")
        assert response.status_code == 200
        
        # Step 3: Verify results
        data = response.json()
        assert len(data) <= 2
        assert all(chunk["id"] in inserted_ids for chunk in data)

    @pytest.mark.asyncio
    async def test_multiple_inserts_and_retrieval(self, client, clear_chunk_store):
        """Test inserting many chunks and retrieving subsets."""
        # Insert 20 chunks
        for i in range(20):
            payload = {
                "text": f"Document {i} about various topics including technology, science, and art.",
                "metadata": {"source": f"doc_{i}.txt", "position": i}
            }
            response = client.post("/v1/chunks/insert", json=payload)
            assert response.status_code == 201
        
        # Retrieve a subset
        response = client.get("/v1/chunks/retrieve?text=technology science&n_chunks=5")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 5
        assert all(isinstance(chunk["metadata"]["position"], int) for chunk in data)
