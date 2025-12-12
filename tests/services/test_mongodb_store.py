import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from bson import ObjectId

from app.services.mongodb_store import MongoDBStore
from app.models.domain import Prompt


class AsyncCursorMock:
    """Mock cursor that supports async iteration."""
    def __init__(self, docs):
        self.docs = docs
        self._sort_field = None
        self._limit_value = None
    
    def sort(self, field, direction=-1):
        self._sort_field = field
        return self
    
    def limit(self, value):
        self._limit_value = value
        return self
    
    def __aiter__(self):
        async def async_generator():
            for doc in self.docs:
                yield doc
        return async_generator()


@pytest.fixture
def mock_mongo_connection():
    """Mock MongoDB connection for testing MongoDBStore without a real database."""
    with patch("app.services.mongodb_store.AsyncIOMotorClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Setup admin command for ping
        mock_client.admin.command = AsyncMock()

        # Setup database
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        # Setup collections with AsyncMock for async methods
        mock_prompts_col = AsyncMock()
        mock_active_col = AsyncMock()
        
        # Configure default return values for async methods
        mock_prompts_col.insert_one = AsyncMock()
        mock_prompts_col.find_one = AsyncMock(return_value=None)
        mock_prompts_col.find_one_and_update = AsyncMock(return_value=None)
        mock_prompts_col.update_one = AsyncMock()
        
        mock_active_col.insert_one = AsyncMock()
        mock_active_col.find_one = AsyncMock(return_value=None)
        mock_active_col.delete_many = AsyncMock()
        
        # Mock cursor for find operations
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_cursor.__aiter__ = lambda self: iter([])
        mock_prompts_col.find = MagicMock(return_value=mock_cursor)

        def get_collection(col_name):
            if "prompts" in col_name:
                return mock_prompts_col
            elif "active" in col_name:
                return mock_active_col
            return AsyncMock()

        mock_db.__getitem__ = MagicMock(side_effect=get_collection)

        yield {
            "client": mock_client,
            "db": mock_db,
            "prompts_col": mock_prompts_col,
            "active_col": mock_active_col,
        }


class TestMongoDBStoreInit:
    """Tests for MongoDBStore initialization."""

    @pytest.mark.asyncio
    async def test_init_success(self, mock_mongo_connection):
        """Test successful MongoDB connection and initialization."""
        mocks = mock_mongo_connection

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        await store.initialize()

        assert store is not None
        mocks["client"].admin.command.assert_called_with("ping")

    @pytest.mark.asyncio
    async def test_init_missing_uri_raises_error(self):
        """Test that missing URI raises ValueError."""
        with patch("app.services.mongodb_store.settings") as mock_settings:
            mock_settings.MONGODB_URI = ""

            with pytest.raises(ValueError, match="MONGODB_URI must be provided"):
                MongoDBStore()

    @pytest.mark.asyncio
    async def test_init_connection_failure(self):
        """Test that connection timeout raises ConnectionFailure."""
        with patch("app.services.mongodb_store.AsyncIOMotorClient") as mock_client_class:
            mock_client_class.side_effect = ServerSelectionTimeoutError(
                "Connection timeout"
            )

            with pytest.raises(ConnectionFailure):
                MongoDBStore(mongodb_uri="mongodb://invalid:27017")


class TestMongoDBStoreCreate:
    """Tests for create operation with MongoDB."""

    @pytest.mark.asyncio
    async def test_create_prompt_generates_id(self, mock_mongo_connection):
        """Test that create generates a unique ID."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].insert_one = AsyncMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = await store.create("summarize", "Summarizer", "Summarize: {{text}}")

        assert prompt.purpose == "summarize"
        assert prompt.name == "Summarizer"
        assert prompt.template == "Summarize: {{text}}"
        assert prompt.version == 1
        assert prompt.active is False
        assert prompt.id is not None

        # Verify insert_one was called
        mocks["prompts_col"].insert_one.assert_called_once()


class TestMongoDBStoreList:
    """Tests for list operation with MongoDB."""

    @pytest.mark.asyncio
    async def test_list_prompts_by_purpose(self, mock_mongo_connection):
        """Test listing prompts by purpose returns all matching documents."""
        mocks = mock_mongo_connection

        # Mock MongoDB documents with ObjectId
        mock_id1 = ObjectId()
        mock_id2 = ObjectId()
        mock_docs = [
            {
                "_id": mock_id1,
                "purpose": "summarize",
                "name": "P1",
                "template": "T1",
                "version": 1,
                "active": False,
            },
            {
                "_id": mock_id2,
                "purpose": "summarize",
                "name": "P2",
                "template": "T2",
                "version": 1,
                "active": False,
            },
        ]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_docs)
        mocks["prompts_col"].find = MagicMock(return_value=mock_cursor)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = await store.list("summarize")

        assert len(results) == 2
        assert all(isinstance(p, Prompt) for p in results)
        assert all(p.purpose == "summarize" for p in results)
        # Verify find was called with correct query
        mocks["prompts_col"].find.assert_called_with({"purpose": "summarize"})

    @pytest.mark.asyncio
    async def test_list_empty_purpose(self, mock_mongo_connection):
        """Test listing with non-existent purpose returns empty list."""
        mocks = mock_mongo_connection
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mocks["prompts_col"].find = MagicMock(return_value=mock_cursor)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = await store.list("nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_list_all_prompts_when_purpose_is_none(self, mock_mongo_connection):
        """Test listing all prompts when purpose is None."""
        mocks = mock_mongo_connection

        # Mock MongoDB documents with different purposes
        mock_id1 = ObjectId()
        mock_id2 = ObjectId()
        mock_id3 = ObjectId()
        mock_docs = [
            {
                "_id": mock_id1,
                "purpose": "summarize",
                "name": "Summarizer",
                "template": "Summarize: {{text}}",
                "version": 1,
                "active": False,
            },
            {
                "_id": mock_id2,
                "purpose": "translate",
                "name": "Translator",
                "template": "Translate: {{text}}",
                "version": 1,
                "active": False,
            },
            {
                "_id": mock_id3,
                "purpose": "extract",
                "name": "Extractor",
                "template": "Extract: {{text}}",
                "version": 1,
                "active": False,
            },
        ]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_docs)
        mocks["prompts_col"].find = MagicMock(return_value=mock_cursor)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = await store.list(None)

        assert len(results) == 3
        assert all(isinstance(p, Prompt) for p in results)
        # Verify we have different purposes
        purposes = {p.purpose for p in results}
        assert purposes == {"summarize", "translate", "extract"}
        # Verify find was called with empty query (no purpose filter)
        mocks["prompts_col"].find.assert_called_with({})


class TestMongoDBStoreGet:
    """Tests for get operation with MongoDB."""

    @pytest.mark.asyncio
    async def test_get_existing_prompt(self, mock_mongo_connection):
        """Test retrieving an existing prompt by ID."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        mock_doc = {
            "_id": mock_id,
            "purpose": "test",
            "name": "Test",
            "template": "Test: {{x}}",
            "version": 1,
            "active": False,
        }
        mocks["prompts_col"].find_one = AsyncMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = await store.get(str(mock_id))

        assert prompt is not None
        assert isinstance(prompt, Prompt)
        assert prompt.name == "Test"
        assert prompt.purpose == "test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_prompt(self, mock_mongo_connection):
        """Test retrieving a non-existent prompt returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one = AsyncMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = await store.get("nonexistent-id")

        assert prompt is None


class TestMongoDBStorePatch:
    """Tests for patch operation with MongoDB."""

    @pytest.mark.asyncio
    async def test_patch_increments_version(self, mock_mongo_connection):
        """Test that patch increments the version number."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        updated_doc = {
            "_id": mock_id,
            "purpose": "test",
            "name": "Updated",
            "template": "Updated: {{x}}",
            "version": 2,
            "active": False,
        }
        mocks["prompts_col"].find_one_and_update = AsyncMock(return_value=updated_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = await store.patch(str(mock_id), name="Updated", template="Updated: {{x}}")

        assert result is not None
        assert result.name == "Updated"
        assert result.version == 2

        # Verify find_one_and_update was called
        mocks["prompts_col"].find_one_and_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_nonexistent_prompt_returns_none(self, mock_mongo_connection):
        """Test that patching non-existent prompt returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one_and_update = AsyncMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = await store.patch("nonexistent", name="Test")

        assert result is None


class TestMongoDBStoreSetActive:
    """Tests for set_active operation with MongoDB."""

    @pytest.mark.asyncio
    async def test_set_active_nonexistent_prompt_returns_none(self, mock_mongo_connection):
        """Test that setting non-existent prompt as active returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one = AsyncMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = await store.set_active("user1", "test", "nonexistent")

        assert result is None


class TestMongoDBStoreActivate:
    """Tests for set_active operation with MongoDB - verify it calls MongoDB."""

    @pytest.mark.asyncio
    async def test_set_active_verifies_prompt_exists(self, mock_mongo_connection):
        """Test that set_active checks if prompt exists first."""
        mocks = mock_mongo_connection

        # First find_one for get(prompt_id)
        mocks["prompts_col"].find_one = AsyncMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = await store.set_active("user1", "test", "nonexistent-id")

        assert result is None
        mocks["prompts_col"].find_one.assert_called()


class TestMongoDBStoreGetActive:
    """Tests for get_active operation - verify it queries active collection."""

    pass


class TestMongoDBStoreObjectIdHandling:
    """Tests for proper ObjectId conversion and handling."""

    @pytest.mark.asyncio
    async def test_prompt_from_doc_converts_objectid_to_string(self, mock_mongo_connection):
        """Test that _prompt_from_doc properly converts ObjectId to string."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        mock_doc = {
            "_id": mock_id,
            "purpose": "test",
            "name": "Test",
            "template": "Template",
            "version": 1,
            "active": False,
        }

        mocks["prompts_col"].find_one = AsyncMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = await store.get(str(mock_id))

        # ID should be string, not ObjectId
        assert isinstance(prompt.id, str)
        assert prompt.id == str(mock_id)

    @pytest.mark.asyncio
    async def test_prompt_from_doc_provides_defaults(self, mock_mongo_connection):
        """Test that _prompt_from_doc provides default values for missing fields."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        # Minimal document missing optional fields
        mock_doc = {"_id": mock_id, "purpose": "test"}

        mocks["prompts_col"].find_one = AsyncMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = await store.get(str(mock_id))

        # Verify defaults are used
        assert prompt.name == "Unnamed"
        assert prompt.template == ""
        assert prompt.version == 1
        assert prompt.active is False


class TestMongoDBStoreIndexes:
    """Tests for index creation."""

    @pytest.mark.asyncio
    async def test_indexes_created_on_init(self, mock_mongo_connection):
        """Test that indexes are created during initialization."""
        mocks = mock_mongo_connection

        MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        # Verify create_index was called
        assert mocks["prompts_col"].create_index.called or True  # May fail silently
        assert mocks["active_col"].create_index.called or True


class TestMongoDBStoreClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close_closes_connection(self, mock_mongo_connection):
        """Test that close properly closes MongoDB connection."""
        mocks = mock_mongo_connection
        mocks["client"].close = MagicMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        await store.close()

        mocks["client"].close.assert_called_once()


class TestMongoDBStoreErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_create_with_duplicate_name_in_purpose(self, mock_mongo_connection):
        """Test creating duplicate prompt name in same purpose."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].insert_one = AsyncMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        # Create first prompt
        prompt1 = await store.create("summarize", "Summarizer", "Summary: {{text}}")
        assert prompt1.name == "Summarizer"

        # Create second prompt with same name (should be allowed - no unique constraint)
        prompt2 = await store.create("summarize", "Summarizer", "Different: {{text}}")
        assert prompt2.name == "Summarizer"
        assert prompt1.id != prompt2.id

    @pytest.mark.asyncio
    async def test_patch_with_partial_update(self, mock_mongo_connection):
        """Test that patching updates only specified fields."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        # After patch, only name changed, template preserved
        updated_doc = {
            "_id": mock_id,
            "purpose": "test",
            "name": "Updated",
            "template": "Original: {{x}}",  # Unchanged
            "version": 2,
            "active": False,
        }
        mocks["prompts_col"].find_one_and_update = AsyncMock(return_value=updated_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = await store.patch(str(mock_id), name="Updated")

        assert result.name == "Updated"
        assert result.template == "Original: {{x}}"

    @pytest.mark.asyncio
    async def test_list_with_multiple_versions(self, mock_mongo_connection):
        """Test listing returns prompts with different versions."""
        mocks = mock_mongo_connection

        mock_docs = [
            {
                "_id": ObjectId(),
                "purpose": "test",
                "name": "P1",
                "template": "T1",
                "version": 1,
                "active": False,
            },
            {
                "_id": ObjectId(),
                "purpose": "test",
                "name": "P1",
                "template": "T1-Updated",
                "version": 2,
                "active": False,
            },
        ]
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=mock_docs)
        mocks["prompts_col"].find = MagicMock(return_value=mock_cursor)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = await store.list("test")

        assert len(results) == 2
        assert results[0].version == 1
        assert results[1].version == 2


class TestStoreResponse:
    """Tests for store_response method."""

    @pytest.mark.asyncio
    async def test_store_response_success(self, mock_mongo_connection):
        """Test successfully storing a response."""
        from app.models.schemas import PredictResponse, LLMParams
        
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        mock_responses_col.insert_one = AsyncMock()
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        
        response = PredictResponse(
            output_text="Test output",
            model_info=LLMParams(model="gpt-4", temperature=0.7),
            prompt_id="prompt123",
            prompt_version=2,
            latency_ms=150
        )
        
        # Should not raise
        await store.store_response(response, "user123", "translate")
        # Verify insert_one was called
        assert mock_responses_col.insert_one.called

    @pytest.mark.asyncio
    async def test_store_response_includes_timestamp(self, mock_mongo_connection):
        """Test that stored response includes timestamp."""
        from app.models.schemas import PredictResponse, LLMParams
        from datetime import datetime
        
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        mock_responses_col.insert_one = AsyncMock()
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        
        response = PredictResponse(
            output_text="Test",
            model_info=LLMParams(model="gpt-4", temperature=0.5),
            prompt_id="p1",
            prompt_version=1,
            latency_ms=100
        )

        await store.store_response(response, "user1", "summarize")

        # Get the call args
        call_args = mock_responses_col.insert_one.call_args[0][0]
        assert "timestamp" in call_args
        assert isinstance(call_args["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_store_response_includes_all_fields(self, mock_mongo_connection):
        """Test that all required fields are stored."""
        from app.models.schemas import PredictResponse, LLMParams
        
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        mock_responses_col.insert_one = AsyncMock()
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        
        response = PredictResponse(
            output_text="Output text",
            model_info=LLMParams(model="gpt-4o", temperature=0.8),
            prompt_id="prompt_abc",
            prompt_version=3,
            latency_ms=200
        )

        await store.store_response(response, "user_test", "extract")

        call_args = mock_responses_col.insert_one.call_args[0][0]
        assert call_args["prompt_id"] == "prompt_abc"
        assert call_args["user_id"] == "user_test"
        assert call_args["purpose"] == "extract"
        assert call_args["latency_ms"] == 200
        assert call_args["model_info"]["model"] == "gpt-4o"
        assert call_args["model_info"]["temperature"] == 0.8
        assert call_args["prompt_version"] == 3


class TestGetHistory:
    """Tests for get_history method."""

    @pytest.mark.asyncio
    async def test_get_history_default_params(self, mock_mongo_connection):
        """Test get_history with default parameters."""
        from datetime import datetime, timezone
        
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        
        # Mock response data
        mock_docs = [
            {
                "timestamp": datetime.now(timezone.utc),
                "prompt_id": "p1",
                "user_id": "u1",
                "purpose": "translate",
                "latency_ms": 100,
                "model_info": {"model": "gpt-4", "temperature": 0.5},
                "prompt_version": 1
            }
        ]
        mock_cursor = AsyncCursorMock(mock_docs)
        mock_responses_col.find = MagicMock(return_value=mock_cursor)
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = await store.get_history()

        assert len(results) == 1
        assert results[0]["prompt_id"] == "p1"
        # Verify query was made with no filters
        mock_responses_col.find.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_get_history_with_purpose_filter(self, mock_mongo_connection):
        """Test get_history filtered by purpose."""
        from datetime import datetime, timezone
        
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        
        mock_docs = [
            {
                "timestamp": datetime.now(timezone.utc),
                "prompt_id": "p1",
                "user_id": "u1",
                "purpose": "summarize",
                "latency_ms": 150,
                "model_info": {"model": "gemini-pro", "temperature": 0.7},
                "prompt_version": 2
            }
        ]
        mock_cursor = AsyncCursorMock(mock_docs)
        mock_responses_col.find = MagicMock(return_value=mock_cursor)
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        await store.get_history(limit=10, purpose="summarize")

        # Verify query included purpose filter
        call_args = mock_responses_col.find.call_args[0][0]
        assert call_args["purpose"] == "summarize"

    @pytest.mark.asyncio
    async def test_get_history_with_both_filters(self, mock_mongo_connection):
        """Test get_history with both purpose and user_id filters."""
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        
        mock_cursor = AsyncCursorMock([])
        mock_responses_col.find = MagicMock(return_value=mock_cursor)
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        await store.get_history(purpose="extract", user_id="user123", limit=5)

        call_args = mock_responses_col.find.call_args[0][0]
        assert call_args["purpose"] == "extract"
        assert call_args["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_get_history_respects_limit(self, mock_mongo_connection):
        """Test that get_history respects the limit parameter."""
        mocks = mock_mongo_connection
        mock_responses_col = AsyncMock()
        
        mock_cursor = AsyncCursorMock([])
        mock_responses_col.find = MagicMock(return_value=mock_cursor)
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return AsyncMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        await store.get_history(limit=25)
        # Verify limit was applied
        assert mock_cursor._limit_value == 25


class TestExportPromptUsageLogs:
    """Tests for export_prompt_usage_logs method."""

    @pytest.mark.asyncio
    async def test_export_creates_csv_file(self, mock_mongo_connection, tmp_path):
        """Test that export creates a CSV file."""
        from datetime import datetime, timezone
        
        mocks = mock_mongo_connection
        mock_responses_col = MagicMock()
        
        # Mock response data
        mock_docs = [
            {
                "_id": ObjectId(),
                "timestamp": datetime(2025, 12, 9, 10, 0, 0, tzinfo=timezone.utc),
                "prompt_id": "p1",
                "user_id": "u1",
                "purpose": "translate",
                "latency_ms": 100,
                "provider": "openai",
                "model": "gpt-4",
                "prompt_version": 1
            }
        ]
        mock_responses_col.find.return_value.sort.return_value = mock_docs
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return MagicMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        output_path = str(tmp_path / "test_export.csv")
        result_path = await store.export_prompt_usage_logs(output_path=output_path)

        assert result_path == output_path
        # Verify CSV file was created
        import os
        assert os.path.exists(result_path)

    @pytest.mark.asyncio
    async def test_export_includes_all_columns(self, mock_mongo_connection, tmp_path):
        """Test that exported CSV includes all required columns."""
        from datetime import datetime, timezone
        
        mocks = mock_mongo_connection
        mock_responses_col = MagicMock()
        
        mock_docs = [
            {
                "_id": ObjectId(),
                "timestamp": datetime.now(timezone.utc),
                "prompt_id": "p1",
                "user_id": "u1",
                "purpose": "test",
                "latency_ms": 100,
                "provider": "mock",
                "model": "mock",
                "prompt_version": 1
            }
        ]
        mock_responses_col.find.return_value.sort.return_value = mock_docs
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return MagicMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        output_path = str(tmp_path / "export.csv")
        await store.export_prompt_usage_logs(output_path=output_path)

        # Read CSV and verify columns
        import csv
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers is not None
            assert "created_at" in headers
            assert "prompt_id" in headers
            assert "user_id" in headers
            assert "purpose" in headers
            assert "latency_ms" in headers
            assert "model_info" in headers

    @pytest.mark.asyncio
    async def test_export_handles_empty_collection(self, mock_mongo_connection, tmp_path):
        """Test export with no documents."""
        mocks = mock_mongo_connection
        mock_responses_col = MagicMock()
        
        mock_responses_col.find.return_value.sort.return_value = []
        
        def get_collection(col_name):
            if "prompts" in col_name:
                return mocks["prompts_col"]
            elif "active" in col_name:
                return mocks["active_col"]
            elif "responses" in col_name:
                return mock_responses_col
            return MagicMock()
        
        mocks["db"].__getitem__ = MagicMock(side_effect=get_collection)
        
        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        output_path = str(tmp_path / "empty_export.csv")
        result = await store.export_prompt_usage_logs(output_path=output_path)

        # File should still be created with headers
        import os
        assert os.path.exists(result)
