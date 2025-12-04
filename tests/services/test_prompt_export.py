import pytest
import os
import csv
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from app.services.prompt_export import PromptExportService


@pytest.fixture
def mock_mongo_connection():
    """Mock MongoDB connection for testing PromptExportService."""
    with patch("app.services.prompt_export.MongoClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup database
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        # Setup responses collection
        mock_responses_col = MagicMock()
        mock_logs_col = MagicMock()

        def get_collection(col_name):
            if "responses" in col_name:
                return mock_responses_col
            elif "logs" in col_name:
                return mock_logs_col
            return MagicMock()

        mock_db.__getitem__ = MagicMock(side_effect=get_collection)

        yield {
            "client": mock_client,
            "db": mock_db,
            "responses_col": mock_responses_col,
            "logs_col": mock_logs_col,
        }


class TestPromptExportServiceInit:
    """Tests for PromptExportService initialization."""

    def test_init_success(self, mock_mongo_connection):
        """Test successful MongoDB connection and initialization."""
        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")

        assert service is not None
        assert service.responses_col is not None

    def test_init_missing_uri_raises_error(self):
        """Test that missing URI raises ValueError."""
        with patch("app.services.prompt_export.global_settings") as mock_settings:
            mock_settings.MONGODB_URI = ""

            with pytest.raises(ValueError, match="MONGODB_URI must be provided"):
                PromptExportService()


class TestPromptExportServiceExport:
    """Tests for export_prompt_usage_logs operation."""

    def test_export_creates_csv_with_correct_headers(self, mock_mongo_connection, tmp_path):
        """Test that export creates CSV with correct headers."""
        mocks = mock_mongo_connection

        # Mock empty responses (just to verify structure)
        mocks["responses_col"].find = MagicMock(
            return_value=MagicMock(sort=MagicMock(return_value=[]))
        )

        output_file = os.path.join(str(tmp_path), "test.csv")

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=output_file)

        assert os.path.exists(result_path)

        # Verify CSV headers
        with open(result_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_headers = [
                "created_at",
                "prompt_id",
                "user_id",
                "purpose",
                "latency_ms",
                "model_info",
            ]
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == expected_headers

    def test_export_writes_response_documents_to_csv(self, mock_mongo_connection, tmp_path):
        """Test that export correctly writes response documents to CSV."""
        mocks = mock_mongo_connection

        # Mock response documents
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                "purpose": "summarize",
                "latency_ms": 1500,
                "model_info": {"model": "gemini-2.5-flash", "temperature": 0.7},
                "prompt_version": 1,
                "output_text": "Summary text here",
            },
            {
                "timestamp": datetime(2025, 12, 4, 11, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-2",
                "user_id": "user2",
                "purpose": "translate",
                "latency_ms": 2000,
                "model_info": {"model": "gpt-4", "temperature": 0.5},
                "prompt_version": 2,
                "output_text": "Translated text",
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_docs)
        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        output_file = os.path.join(str(tmp_path), "test.csv")

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=output_file)

        assert os.path.exists(result_path)

        # Verify CSV content
        with open(result_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

            # Verify first row
            assert rows[0]["prompt_id"] == "prompt-1"
            assert rows[0]["user_id"] == "user1"
            assert rows[0]["purpose"] == "summarize"
            assert rows[0]["latency_ms"] == "1500"
            assert "gemini-2.5-flash" in rows[0]["model_info"]
            assert "temp=0.7" in rows[0]["model_info"]

            # Verify second row
            assert rows[1]["prompt_id"] == "prompt-2"
            assert rows[1]["user_id"] == "user2"
            assert rows[1]["purpose"] == "translate"

    def test_export_handles_missing_fields(self, mock_mongo_connection, tmp_path):
        """Test that export handles documents with missing fields gracefully."""
        mocks = mock_mongo_connection

        # Mock documents with missing fields
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                # Missing purpose, latency_ms, model_info
                "prompt_version": 1,
                "output_text": "Text",
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_docs)
        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        output_file = os.path.join(str(tmp_path), "test.csv")

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=output_file)

        # Verify CSV still created successfully
        assert os.path.exists(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["purpose"] == ""
            assert rows[0]["model_info"] == "unknown;temp=N/A"

    def test_export_converts_timestamp_to_isoformat(self, mock_mongo_connection, tmp_path):
        """Test that export converts datetime to ISO format string."""
        mocks = mock_mongo_connection

        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 15, 30, 45, 123000, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                "purpose": "summarize",
                "latency_ms": 1500,
                "model_info": {"model": "gemini-2.5-flash"},
                "prompt_version": 1,
                "output_text": "Text",
            },
        ]

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_docs)
        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        output_file = os.path.join(str(tmp_path), "test.csv")

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=output_file)

        with open(result_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Verify timestamp is in ISO format
            assert "2025-12-04T15:30:45" in rows[0]["created_at"]

    def test_export_creates_output_directory_if_missing(self, mock_mongo_connection, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=[])
        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        # Use nested path that doesn't exist
        output_file = os.path.join(
            str(tmp_path), "nested", "dir", "structure", "test.csv"
        )

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=output_file)

        # Verify directory was created and file exists
        assert os.path.exists(result_path)
        assert os.path.isdir(os.path.dirname(result_path))

    def test_export_uses_default_path_when_none_provided(
        self, mock_mongo_connection
    ):
        """Test that export uses default path when None is provided."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=[])
        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        result_path = service.export_prompt_usage_logs(output_path=None)

        # Verify default path is used
        assert "prompt_logs.csv" in result_path
        assert "exports" in result_path


class TestPromptExportServiceClose:
    """Tests for close operation."""

    def test_close_calls_client_close(self, mock_mongo_connection):
        """Test that close method calls client.close()."""
        mocks = mock_mongo_connection

        service = PromptExportService(mongodb_uri="mongodb://localhost:27017/testdb")
        service.close()

        mocks["client"].close.assert_called_once()
