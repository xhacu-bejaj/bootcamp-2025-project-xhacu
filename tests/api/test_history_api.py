"""
Tests for the /v1/history endpoint.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from app.main import app
from app.api.dependencies import get_store
from app.services.mongodb_store import MongoDBStore

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


class TestHistoryAPI:
    """Tests for GET /v1/history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_default_params(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test GET /v1/history with default parameters."""
        mock_store_override.get_history.return_value = [
            {
                "timestamp": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1", "user_id": "user1", "purpose": "test",
                "latency_ms": 100, "provider": "mock", "model": "mock", "prompt_version": 1,
            }
        ]

        response = await client.get("/v1/history")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["prompt_id"] == "prompt-1"
        assert "timestamp" in data[0]
        
        mock_store_override.get_history.assert_called_once_with(limit=50, purpose=None, user_id=None)

    @pytest.mark.asyncio
    async def test_get_history_with_filters(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test GET /v1/history with query filters."""
        mock_store_override.get_history.return_value = []

        await client.get("/v1/history?limit=25&purpose=summarize&user_id=test_user")

        mock_store_override.get_history.assert_called_once_with(
            limit=25, purpose="summarize", user_id="test_user"
        )

    @pytest.mark.asyncio
    async def test_get_history_empty_string_filters_are_none(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test GET /v1/history with empty string filters which should be treated as None."""
        mock_store_override.get_history.return_value = []
        
        await client.get("/v1/history?purpose=&user_id=")
        
        mock_store_override.get_history.assert_called_once_with(
            limit=50, purpose=None, user_id=None
        )

    @pytest.mark.asyncio
    async def test_get_history_limit_validation(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test that the limit parameter is validated."""
        # This test does not need the mock store as it fails at the validation layer
        response_too_low = await client.get("/v1/history?limit=0")
        assert response_too_low.status_code == 422

        response_too_high = await client.get("/v1/history?limit=1001")
        assert response_too_high.status_code == 422

    @pytest.mark.asyncio
    async def test_get_history_returns_empty_list(self, client: AsyncClient, mock_store_override: AsyncMock):
        """Test that the endpoint returns an empty list when the store has no history."""
        mock_store_override.get_history.return_value = []
        
        response = await client.get("/v1/history")
        
        assert response.status_code == 200
        assert response.json() == []