import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from app.services.response_store import ResponseStore
from app.models.schemas import PredictResponse, ModelInfo


@pytest.fixture
def mock_mongo_connection():
    """Mock MongoDB connection for testing ResponseStore without a real database."""
    with patch("app.services.response_store.MongoClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup database
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        # Setup responses collection
        mock_responses_col = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_responses_col)

        yield {
            "client": mock_client,
            "db": mock_db,
            "responses_col": mock_responses_col,
        }


class TestResponseStoreInit:
    """Tests for ResponseStore initialization."""

    def test_init_success(self, mock_mongo_connection):
        """Test successful MongoDB connection and initialization."""
        mocks = mock_mongo_connection

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")

        assert store is not None
        assert store.responses_col is not None
        # Verify indexes were created
        mocks["responses_col"].create_index.assert_called()

    def test_init_missing_uri_raises_error(self):
        """Test that missing URI raises ValueError."""
        with patch("app.services.response_store.global_settings") as mock_settings:
            mock_settings.MONGODB_URI = ""

            with pytest.raises(ValueError, match="MONGODB_URI must be provided"):
                ResponseStore()


class TestResponseStoreStoreResponse:
    """Tests for store_response operation."""

    def test_store_response_inserts_document(self, mock_mongo_connection):
        """Test that store_response inserts a document with correct fields."""
        mocks = mock_mongo_connection
        mocks["responses_col"].insert_one = MagicMock()

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")

        # Create test response
        model_info = ModelInfo(model="gemini-2.5-flash", temperature=0.7)
        response = PredictResponse(
            output_text="Test output",
            model_info=model_info,
            prompt_id="test-prompt-id",
            prompt_version=1,
            latency_ms=1500,
        )

        store.store_response(response, user_id="test_user", purpose="summarize")

        # Verify insert_one was called
        mocks["responses_col"].insert_one.assert_called_once()

        # Verify document structure
        call_args = mocks["responses_col"].insert_one.call_args[0][0]
        assert call_args["prompt_id"] == "test-prompt-id"
        assert call_args["user_id"] == "test_user"
        assert call_args["purpose"] == "summarize"
        assert call_args["output_text"] == "Test output"
        assert call_args["latency_ms"] == 1500
        assert call_args["prompt_version"] == 1
        assert "timestamp" in call_args
        assert "model_info" in call_args


class TestResponseStoreGetHistory:
    """Tests for get_history operation."""

    def test_get_history_no_filters(self, mock_mongo_connection):
        """Test get_history without filters returns all records."""
        mocks = mock_mongo_connection

        # Mock cursor and results
        mock_cursor = MagicMock()
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                "purpose": "summarize",
                "latency_ms": 1000,
                "model_info": {"model": "gemini-2.5-flash", "temperature": 0.7},
                "prompt_version": 1,
            },
            {
                "timestamp": datetime(2025, 12, 4, 11, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-2",
                "user_id": "user2",
                "purpose": "translate",
                "latency_ms": 2000,
                "model_info": {"model": "gpt-4", "temperature": 0.5},
                "prompt_version": 2,
            },
        ]
        mock_cursor.__iter__ = MagicMock(return_value=iter(mock_docs))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.get_history(limit=50)

        # Verify query and sorting
        mocks["responses_col"].find.assert_called_once_with({})
        mock_cursor.sort.assert_called_once_with("timestamp", -1)
        mock_cursor.limit.assert_called_once_with(50)

        # Verify results
        assert len(results) == 2
        assert results[0]["prompt_id"] == "prompt-1"
        assert results[0]["user_id"] == "user1"
        assert results[0]["purpose"] == "summarize"
        assert results[0]["latency_ms"] == 1000
        assert results[0]["provider"] == "gemini-2.5-flash"
        assert results[0]["model"] == "gemini-2.5-flash"
        assert results[1]["prompt_id"] == "prompt-2"

    def test_get_history_with_purpose_filter(self, mock_mongo_connection):
        """Test get_history with purpose filter."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                "purpose": "summarize",
                "latency_ms": 1000,
                "model_info": {"model": "gemini-2.5-flash", "temperature": 0.7},
                "prompt_version": 1,
            },
        ]
        mock_cursor.__iter__ = MagicMock(return_value=iter(mock_docs))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.get_history(limit=50, purpose="summarize")

        # Verify query includes purpose filter
        mocks["responses_col"].find.assert_called_once_with({"purpose": "summarize"})
        assert len(results) == 1
        assert results[0]["purpose"] == "summarize"

    def test_get_history_with_user_id_filter(self, mock_mongo_connection):
        """Test get_history with user_id filter."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "test_user",
                "purpose": "summarize",
                "latency_ms": 1000,
                "model_info": {"model": "gemini-2.5-flash", "temperature": 0.7},
                "prompt_version": 1,
            },
        ]
        mock_cursor.__iter__ = MagicMock(return_value=iter(mock_docs))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.get_history(limit=50, user_id="test_user")

        # Verify query includes user_id filter
        mocks["responses_col"].find.assert_called_once_with({"user_id": "test_user"})
        assert len(results) == 1
        assert results[0]["user_id"] == "test_user"

    def test_get_history_with_both_filters(self, mock_mongo_connection):
        """Test get_history with both purpose and user_id filters."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_docs = []
        mock_cursor.__iter__ = MagicMock(return_value=iter(mock_docs))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.get_history(
            limit=10, purpose="summarize", user_id="test_user"
        )

        # Verify query includes both filters
        mocks["responses_col"].find.assert_called_once_with(
            {"purpose": "summarize", "user_id": "test_user"}
        )
        mock_cursor.limit.assert_called_once_with(10)
        assert len(results) == 0

    def test_get_history_custom_limit(self, mock_mongo_connection):
        """Test get_history respects custom limit."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        store.get_history(limit=100)

        mock_cursor.limit.assert_called_once_with(100)

    def test_get_history_handles_missing_model_info(self, mock_mongo_connection):
        """Test get_history handles documents with missing model info."""
        mocks = mock_mongo_connection

        mock_cursor = MagicMock()
        mock_docs = [
            {
                "timestamp": datetime(2025, 12, 4, 12, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "prompt-1",
                "user_id": "user1",
                "purpose": "summarize",
                "latency_ms": 1000,
                "model_info": {},  # Empty model_info
                "prompt_version": 1,
            },
        ]
        mock_cursor.__iter__ = MagicMock(return_value=iter(mock_docs))
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)

        mocks["responses_col"].find = MagicMock(return_value=mock_cursor)

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.get_history(limit=50)

        # Verify it returns "unknown" for missing model
        assert len(results) == 1
        assert results[0]["provider"] == "unknown"
        assert results[0]["model"] == "unknown"


class TestResponseStoreClose:
    """Tests for close operation."""

    def test_close_calls_client_close(self, mock_mongo_connection):
        """Test that close method calls client.close()."""
        mocks = mock_mongo_connection

        store = ResponseStore(mongodb_uri="mongodb://localhost:27017/testdb")
        store.close()

        mocks["client"].close.assert_called_once()
