from unittest.mock import patch, create_autospec
from app.services.mongodb_store import MongoDBStore


class TestPromptExportAPI:
    """Tests for POST /v1/prompts/export endpoint."""

    def test_export_prompts_success(self, client):
        """Test POST /v1/prompts/export returns success response."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["export_path"] == "var/exports/prompt_logs.csv"
            assert "message" in data
            assert "exported successfully" in data["message"]

            mock_store.export_prompt_usage_logs.assert_called_once()

    def test_export_prompts_with_header(self, client):
        """Test POST /v1/prompts/export works with x_user_id header."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post(
                "/v1/prompts/export", headers={"X-User-Id": "admin_user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    def test_export_prompts_missing_mongodb_uri(self, client):
        """Test POST /v1/prompts/export handles missing MongoDB store."""
        from app.services.prompt_store import InMemoryStore
        mock_store = create_autospec(InMemoryStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            response = client.post("/v1/prompts/export")

            assert response.status_code == 400
            data = response.json()
            assert "Export requires MongoDB store" in data["detail"]

    def test_export_prompts_io_error(self, client):
        """Test POST /v1/prompts/export handles IO errors."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.side_effect = IOError(
                "Cannot write to file"
            )

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Export failed" in data["detail"]

    def test_export_prompts_unexpected_error(self, client):
        """Test POST /v1/prompts/export handles unexpected errors."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.side_effect = Exception("Database connection lost")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Unexpected error" in data["detail"]

    def test_export_prompts_service_close_on_success(self, client):
        """Test that shared store doesn't need per-request close."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200

    def test_export_prompts_service_close_on_io_error(self, client):
        """Test that shared store doesn't need per-request close even on error."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
            mock_store.export_prompt_usage_logs.side_effect = IOError("File error")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500

    def test_export_prompts_response_format(self, client):
        """Test that export response has correct format."""
        mock_store = create_autospec(MongoDBStore, instance=True)
        with patch("app.api.routes_prompts.store", mock_store):
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
