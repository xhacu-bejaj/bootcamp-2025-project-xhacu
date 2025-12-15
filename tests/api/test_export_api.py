"""
Tests for the /v1/prompts/export endpoint.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock

from app.main import app
from app.api.dependencies import get_store
from app.services.mongodb_store import MongoDBStore
from app.services.prompt_store import InMemoryStore

@pytest.fixture
def mock_store_override():
    """
    Fixture to create and override the get_store dependency with a mock.
    Yields the mock store so tests can configure its return values.
    """
    mock_store = AsyncMock(spec=MongoDBStore)
    app.dependency_overrides[get_store] = lambda: mock_store
    yield mock_store
    app.dependency_overrides.clear()


class TestPromptExportAPI:
    """Tests for POST /v1/prompts/export endpoint."""

    @pytest.mark.asyncio
    async def test_export_prompts_success(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test POST /v1/prompts/export returns a success response."""
        expected_path = "var/exports/prompt_logs.csv"
        mock_store_override.export_prompt_usage_logs.return_value = expected_path

        response = await client.post("/v1/prompts/export")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["export_path"] == expected_path
        mock_store_override.export_prompt_usage_logs.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_fails_with_in_memory_store(self, client: AsyncClient):
        """Test that export fails when not using MongoDB."""
        # Override the dependency specifically for this test
        app.dependency_overrides[get_store] = lambda: InMemoryStore()
        
        response = await client.post("/v1/prompts/export")
        
        assert response.status_code == 400
        assert "Export requires MongoDB store" in response.json()["detail"]
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_handles_io_error(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test that the endpoint handles IO errors during export."""
        mock_store_override.export_prompt_usage_logs.side_effect = IOError("Disk is full")

        response = await client.post("/v1/prompts/export")

        assert response.status_code == 500
        assert "Export failed" in response.json()["detail"]