"""
Tests for the chunks API endpoints.
"""

import pytest
from httpx import AsyncClient
from app.app_context import app_context
from app.services.chunk_store import ChunkStore
from app.core.config import settings

@pytest.fixture(autouse=True)
async def chunk_store_for_api_tests():
    """
    Fixture to initialize and clear the app_context.chunk_store for API tests.
    `autouse=True` ensures it runs for every test in this module.
    """
    if not hasattr(app_context, 'chunk_store') or not app_context.chunk_store:
        app_context.chunk_store = ChunkStore()
    
    # Clear the store before each test
    app_context.chunk_store.clear()
    yield
    # Teardown: Clear after the test is not strictly necessary due to clearing before,
    # but it's good practice for hygiene.
    app_context.chunk_store.clear()


class TestChunkInsertAPI:
    """Tests for POST /v1/chunks/insert endpoint."""

    @pytest.mark.asyncio
    async def test_insert_chunk_success(self, client: AsyncClient):
        """Test successfully inserting a chunk."""
        payload = {
            "text": "This is a test chunk for the API.",
            "metadata": {"source": "test_document.txt", "position": 0}
        }
        response = await client.post("/v1/chunks/insert", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["text"] == payload["text"]

    @pytest.mark.asyncio
    async def test_insert_chunk_exceeds_max_length(self, client: AsyncClient):
        """Test that inserting a chunk exceeding max length returns 400."""
        payload = {"text": "x" * (settings.MAX_CHUNK_LENGTH + 1)}
        response = await client.post("/v1/chunks/insert", json=payload)
        assert response.status_code == 400
        assert "exceeds maximum length" in response.json()["detail"]


class TestChunkBatchInsertAPI:
    """Tests for POST /v1/chunks/insert_batch endpoint."""

    @pytest.mark.asyncio
    async def test_insert_batch_success(self, client: AsyncClient):
        """Test successfully inserting a batch of chunks."""
        payload = {
            "chunks": [
                {"text": "First batch chunk.", "metadata": {"source": "doc1"}},
                {"text": "Second batch chunk.", "metadata": {"source": "doc2"}},
            ]
        }
        response = await client.post("/v1/chunks/insert_batch", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["text"] == "First batch chunk."
        assert data[1]["metadata"]["source"] == "doc2"

    @pytest.mark.asyncio
    async def test_insert_batch_validation_error(self, client: AsyncClient):
        """Test that a validation error in one chunk fails the batch."""
        payload = {
            "chunks": [
                {"text": "This chunk is okay."},
                {"text": "x" * (settings.MAX_CHUNK_LENGTH + 1)},
            ]
        }
        response = await client.post("/v1/chunks/insert_batch", json=payload)
        assert response.status_code == 400
        assert "exceeds maximum length" in response.json()["detail"]


class TestChunkRetrieveAPI:
    """Tests for GET /v1/chunks/retrieve endpoint."""

    @pytest.mark.asyncio
    async def test_retrieve_chunks_success(self, client: AsyncClient):
        """Test successfully retrieving chunks."""
        await client.post("/v1/chunks/insert", json={"text": "Python is a programming language."})
        
        response = await client.get("/v1/chunks/retrieve?text=programming languages")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "Python" in data[0]["text"]

    @pytest.mark.asyncio
    async def test_retrieve_chunks_empty_database(self, client: AsyncClient):
        """Test retrieving from empty database."""
        response = await client.get("/v1/chunks/retrieve?text=anything")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_retrieve_chunks_missing_text_param(self, client: AsyncClient):
        """Test that missing text parameter returns 422."""
        response = await client.get("/v1/chunks/retrieve")
        assert response.status_code == 422