import pytest
from unittest.mock import AsyncMock
from app.services.mongodb_store import MongoDBStore
from app.api.dependencies import get_store
from app.main import app


class TestPromptExportAPI:
    """Tests for POST /v1/prompts/export endpoint."""

    @pytest.mark.asyncio
    async def test_export_prompts_success(self, client):
        """Test POST /v1/prompts/export returns success response."""
        mock_store = AsyncMock(spec=MongoDBStore)
        mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"
        app.dependency_overrides[get_store] = lambda: mock_store
        try:

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["export_path"] == "var/exports/prompt_logs.csv"
            assert "message" in data
            assert "exported successfully" in data["message"]

            mock_store.export_prompt_usage_logs.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_with_header(self, client):
        """Test POST /v1/prompts/export works with x_user_id header."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post(
                "/v1/prompts/export", headers={"X-User-Id": "admin_user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_missing_mongodb_uri(self, client):
        """Test POST /v1/prompts/export handles missing MongoDB store."""
        from app.services.prompt_store import InMemoryStore
        mock_store = InMemoryStore()
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            response = client.post("/v1/prompts/export")

            assert response.status_code == 400
            data = response.json()
            assert "Export requires MongoDB store" in data["detail"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_io_error(self, client):
        """Test POST /v1/prompts/export handles IO errors."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.side_effect = IOError(
                "Cannot write to file"
            )

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Export failed" in data["detail"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_unexpected_error(self, client):
        """Test POST /v1/prompts/export handles unexpected errors."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.side_effect = Exception("Database connection lost")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Unexpected error" in data["detail"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_service_close_on_success(self, client):
        """Test that shared store doesn't need per-request close."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_service_close_on_io_error(self, client):
        """Test that shared store doesn't need per-request close even on error."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.side_effect = IOError("File error")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_export_prompts_response_format(self, client):
        """Test that export response has correct format."""
        mock_store = AsyncMock(spec=MongoDBStore)
        app.dependency_overrides[get_store] = lambda: mock_store
        try:
            mock_store.export_prompt_usage_logs.return_value = (
                "var/exports/prompt_logs.csv"
            )

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "export_path" in data
            assert "message" in data

            assert isinstance(data["status"], str)
            assert isinstance(data["export_path"], str)
            assert isinstance(data["message"], str)
        finally:
            app.dependency_overrides.clear()
