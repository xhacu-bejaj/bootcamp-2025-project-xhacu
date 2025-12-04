from unittest.mock import patch, MagicMock


class TestPromptExportAPI:
    """Tests for POST /v1/prompts/export endpoint."""

    def test_export_prompts_success(self, client):
        """Test POST /v1/prompts/export returns success response."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["export_path"] == "var/exports/prompt_logs.csv"
            assert "message" in data
            assert "exported successfully" in data["message"]

            # Verify service was called and closed
            mock_service.export_prompt_usage_logs.assert_called_once()
            mock_service.close.assert_called_once()

    def test_export_prompts_with_header(self, client):
        """Test POST /v1/prompts/export works with x_user_id header."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post(
                "/v1/prompts/export", headers={"X-User-Id": "admin_user"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    def test_export_prompts_missing_mongodb_uri(self, client):
        """Test POST /v1/prompts/export handles missing MONGODB_URI."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            MockService.side_effect = ValueError("MONGODB_URI must be provided")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 400
            data = response.json()
            assert "Configuration error" in data["detail"]

    def test_export_prompts_io_error(self, client):
        """Test POST /v1/prompts/export handles IO errors."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.side_effect = IOError(
                "Cannot write to file"
            )

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Export failed" in data["detail"]

            # Verify close was still called
            mock_service.close.assert_called_once()

    def test_export_prompts_unexpected_error(self, client):
        """Test POST /v1/prompts/export handles unexpected errors."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            MockService.side_effect = Exception("Database connection lost")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            data = response.json()
            assert "Unexpected error" in data["detail"]

    def test_export_prompts_service_close_on_success(self, client):
        """Test that service.close() is called on success."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.return_value = "var/exports/prompt_logs.csv"

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            mock_service.close.assert_called_once()

    def test_export_prompts_service_close_on_io_error(self, client):
        """Test that service.close() is called even on IO error."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.side_effect = IOError("File error")

            response = client.post("/v1/prompts/export")

            assert response.status_code == 500
            # Verify close was called
            mock_service.close.assert_called_once()

    def test_export_prompts_response_format(self, client):
        """Test that export response has correct format."""
        with patch("app.api.routes_prompts.PromptExportService") as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.export_prompt_usage_logs.return_value = (
                "var/exports/prompt_logs.csv"
            )

            response = client.post("/v1/prompts/export")

            assert response.status_code == 200
            data = response.json()

            # Verify required fields
            assert "status" in data
            assert "export_path" in data
            assert "message" in data

            # Verify types
            assert isinstance(data["status"], str)
            assert isinstance(data["export_path"], str)
            assert isinstance(data["message"], str)
