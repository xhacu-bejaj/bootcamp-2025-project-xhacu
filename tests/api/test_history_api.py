import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


class TestHistoryAPI:
    """Tests for /v1/history endpoint."""

    def test_get_history_default_params(self, client):
        """Test GET /v1/history with default parameters."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store

            # Mock get_history response
            mock_store.get_history.return_value = [
                {
                    "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                    "prompt_id": "prompt-1",
                    "user_id": "user1",
                    "purpose": "summarize",
                    "latency_ms": 1500,
                    "provider": "gemini-2.5-flash",
                    "model": "gemini-2.5-flash",
                    "prompt_version": 1,
                },
                {
                    "timestamp": datetime(2025, 12, 4, 11, 0, 0, tzinfo=timezone.utc),
                    "prompt_id": "prompt-2",
                    "user_id": "user2",
                    "purpose": "translate",
                    "latency_ms": 2000,
                    "provider": "gpt-4",
                    "model": "gpt-4",
                    "prompt_version": 2,
                },
            ]

            response = client.get("/v1/history")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

            # Verify first record
            assert data[0]["prompt_id"] == "prompt-1"
            assert data[0]["user_id"] == "user1"
            assert data[0]["purpose"] == "summarize"
            assert data[0]["latency_ms"] == 1500
            assert data[0]["provider"] == "gemini-2.5-flash"
            assert data[0]["model"] == "gemini-2.5-flash"
            assert data[0]["prompt_version"] == 1
            assert "timestamp" in data[0]

            # Verify store was called with defaults
            mock_store.get_history.assert_called_once_with(
                limit=50, purpose=None, user_id=None
            )
            mock_store.close.assert_called_once()

    def test_get_history_custom_limit(self, client):
        """Test GET /v1/history with custom limit."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            response = client.get("/v1/history?limit=10")

            assert response.status_code == 200
            mock_store.get_history.assert_called_once_with(
                limit=10, purpose=None, user_id=None
            )

    def test_get_history_with_purpose_filter(self, client):
        """Test GET /v1/history with purpose filter."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = [
                {
                    "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                    "prompt_id": "prompt-1",
                    "user_id": "user1",
                    "purpose": "summarize",
                    "latency_ms": 1500,
                    "provider": "gemini-2.5-flash",
                    "model": "gemini-2.5-flash",
                    "prompt_version": 1,
                }
            ]

            response = client.get("/v1/history?purpose=summarize")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["purpose"] == "summarize"

            mock_store.get_history.assert_called_once_with(
                limit=50, purpose="summarize", user_id=None
            )

    def test_get_history_with_user_id_filter(self, client):
        """Test GET /v1/history with user_id filter."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = [
                {
                    "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                    "prompt_id": "prompt-1",
                    "user_id": "test_user",
                    "purpose": "summarize",
                    "latency_ms": 1500,
                    "provider": "gemini-2.5-flash",
                    "model": "gemini-2.5-flash",
                    "prompt_version": 1,
                }
            ]

            response = client.get("/v1/history?user_id=test_user")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["user_id"] == "test_user"

            mock_store.get_history.assert_called_once_with(
                limit=50, purpose=None, user_id="test_user"
            )

    def test_get_history_with_both_filters(self, client):
        """Test GET /v1/history with both purpose and user_id filters."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            response = client.get(
                "/v1/history?purpose=summarize&user_id=test_user&limit=25"
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 0

            mock_store.get_history.assert_called_once_with(
                limit=25, purpose="summarize", user_id="test_user"
            )

    def test_get_history_with_empty_string_filters(self, client):
        """Test GET /v1/history with empty string filters (converted to None)."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            response = client.get("/v1/history?purpose=&user_id=")

            assert response.status_code == 200

            # Empty strings should be converted to None
            mock_store.get_history.assert_called_once_with(
                limit=50, purpose=None, user_id=None
            )

    def test_get_history_limit_validation_min(self, client):
        """Test GET /v1/history rejects limit < 1."""
        response = client.get("/v1/history?limit=0")

        assert response.status_code == 422  # Validation error

    def test_get_history_limit_validation_max(self, client):
        """Test GET /v1/history rejects limit > 1000."""
        response = client.get("/v1/history?limit=1001")

        assert response.status_code == 422  # Validation error

    def test_get_history_limit_boundary_values(self, client):
        """Test GET /v1/history accepts boundary values for limit."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            # Test minimum valid limit
            response = client.get("/v1/history?limit=1")
            assert response.status_code == 200
            mock_store.get_history.assert_called_with(
                limit=1, purpose=None, user_id=None
            )

            # Test maximum valid limit
            response = client.get("/v1/history?limit=1000")
            assert response.status_code == 200
            mock_store.get_history.assert_called_with(
                limit=1000, purpose=None, user_id=None
            )

    def test_get_history_returns_empty_list_when_no_data(self, client):
        """Test GET /v1/history returns empty list when no data exists."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            response = client.get("/v1/history")

            assert response.status_code == 200
            data = response.json()
            assert data == []
            assert isinstance(data, list)

    def test_get_history_converts_datetime_to_iso_string(self, client):
        """Test that datetime objects are converted to ISO format strings."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store

            # Return datetime object (not string)
            mock_store.get_history.return_value = [
                {
                    "timestamp": datetime(2025, 12, 4, 12, 30, 45, tzinfo=timezone.utc),
                    "prompt_id": "prompt-1",
                    "user_id": "user1",
                    "purpose": "summarize",
                    "latency_ms": 1500,
                    "provider": "gemini-2.5-flash",
                    "model": "gemini-2.5-flash",
                    "prompt_version": 1,
                }
            ]

            response = client.get("/v1/history")

            assert response.status_code == 200
            data = response.json()
            # Verify timestamp is a string in ISO format
            assert isinstance(data[0]["timestamp"], str)
            assert "2025-12-04T12:30:45" in data[0]["timestamp"]

    def test_get_history_store_close_called_on_success(self, client):
        """Test that store.close() is called even on success."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.return_value = []

            response = client.get("/v1/history")

            assert response.status_code == 200
            mock_store.close.assert_called_once()

    def test_get_history_store_close_called_on_error(self, client):
        """Test that store.close() is called even if get_history raises error."""
        with patch("app.api.routes_history.ResponseStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mock_store.get_history.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                client.get("/v1/history")

            # Verify close was still called due to finally block
            mock_store.close.assert_called_once()
