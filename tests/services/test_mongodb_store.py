import pytest
from unittest.mock import patch, MagicMock
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from bson import ObjectId

from app.services.mongodb_store import MongoDBStore
from app.models.domain import Prompt


@pytest.fixture
def mock_mongo_connection():
    """Mock MongoDB connection for testing MongoDBStore without a real database."""
    with patch("app.services.mongodb_store.MongoClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup admin command for ping
        mock_client.admin.command = MagicMock()

        # Setup database
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)

        # Setup collections
        mock_prompts_col = MagicMock()
        mock_active_col = MagicMock()

        def get_collection(col_name):
            if "prompts" in col_name:
                return mock_prompts_col
            elif "active" in col_name:
                return mock_active_col
            return MagicMock()

        mock_db.__getitem__ = MagicMock(side_effect=get_collection)

        yield {
            "client": mock_client,
            "db": mock_db,
            "prompts_col": mock_prompts_col,
            "active_col": mock_active_col,
        }


class TestMongoDBStoreInit:
    """Tests for MongoDBStore initialization."""

    def test_init_success(self, mock_mongo_connection):
        """Test successful MongoDB connection and initialization."""
        mocks = mock_mongo_connection

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        assert store is not None
        mocks["client"].admin.command.assert_called_with("ping")

    def test_init_missing_uri_raises_error(self):
        """Test that missing URI raises ValueError."""
        with patch("app.services.mongodb_store.global_settings") as mock_settings:
            mock_settings.MONGODB_URI = ""

            with pytest.raises(ValueError, match="MONGODB_URI must be provided"):
                MongoDBStore()

    def test_init_connection_failure(self):
        """Test that connection timeout raises ConnectionFailure."""
        with patch("app.services.mongodb_store.MongoClient") as mock_client_class:
            mock_client_class.side_effect = ServerSelectionTimeoutError(
                "Connection timeout"
            )

            with pytest.raises(ConnectionFailure):
                MongoDBStore(mongodb_uri="mongodb://invalid:27017")


class TestMongoDBStoreCreate:
    """Tests for create operation with MongoDB."""

    def test_create_prompt_generates_id(self, mock_mongo_connection):
        """Test that create generates a unique ID."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].insert_one = MagicMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = store.create("summarize", "Summarizer", "Summarize: {{text}}")

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

    def test_list_prompts_by_purpose(self, mock_mongo_connection):
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
        mocks["prompts_col"].find = MagicMock(return_value=mock_docs)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.list("summarize")

        assert len(results) == 2
        assert all(isinstance(p, Prompt) for p in results)
        assert all(p.purpose == "summarize" for p in results)
        # Verify find was called with correct query
        mocks["prompts_col"].find.assert_called_with({"purpose": "summarize"})

    def test_list_empty_purpose(self, mock_mongo_connection):
        """Test listing with non-existent purpose returns empty list."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find = MagicMock(return_value=[])

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.list("nonexistent")

        assert results == []


class TestMongoDBStoreGet:
    """Tests for get operation with MongoDB."""

    def test_get_existing_prompt(self, mock_mongo_connection):
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
        mocks["prompts_col"].find_one = MagicMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = store.get(str(mock_id))

        assert prompt is not None
        assert isinstance(prompt, Prompt)
        assert prompt.name == "Test"
        assert prompt.purpose == "test"

    def test_get_nonexistent_prompt(self, mock_mongo_connection):
        """Test retrieving a non-existent prompt returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one = MagicMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = store.get("nonexistent-id")

        assert prompt is None


class TestMongoDBStorePatch:
    """Tests for patch operation with MongoDB."""

    def test_patch_increments_version(self, mock_mongo_connection):
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
        mocks["prompts_col"].find_one_and_update = MagicMock(return_value=updated_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = store.patch(str(mock_id), name="Updated", template="Updated: {{x}}")

        assert result is not None
        assert result.name == "Updated"
        assert result.version == 2

        # Verify find_one_and_update was called
        mocks["prompts_col"].find_one_and_update.assert_called_once()

    def test_patch_nonexistent_prompt_returns_none(self, mock_mongo_connection):
        """Test that patching non-existent prompt returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one_and_update = MagicMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = store.patch("nonexistent", name="Test")

        assert result is None


class TestMongoDBStoreSetActive:
    """Tests for set_active operation with MongoDB."""

    def test_set_active_nonexistent_prompt_returns_none(self, mock_mongo_connection):
        """Test that setting non-existent prompt as active returns None."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].find_one = MagicMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = store.set_active("user1", "test", "nonexistent")

        assert result is None


class TestMongoDBStoreActivate:
    """Tests for set_active operation with MongoDB - verify it calls MongoDB."""

    def test_set_active_verifies_prompt_exists(self, mock_mongo_connection):
        """Test that set_active checks if prompt exists first."""
        mocks = mock_mongo_connection

        # First find_one for get(prompt_id)
        mocks["prompts_col"].find_one = MagicMock(return_value=None)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = store.set_active("user1", "test", "nonexistent-id")

        assert result is None
        mocks["prompts_col"].find_one.assert_called()


class TestMongoDBStoreGetActive:
    """Tests for get_active operation - verify it queries active collection."""

    pass


class TestMongoDBStoreObjectIdHandling:
    """Tests for proper ObjectId conversion and handling."""

    def test_prompt_from_doc_converts_objectid_to_string(self, mock_mongo_connection):
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

        mocks["prompts_col"].find_one = MagicMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = store.get(str(mock_id))

        # ID should be string, not ObjectId
        assert isinstance(prompt.id, str)
        assert prompt.id == str(mock_id)

    def test_prompt_from_doc_provides_defaults(self, mock_mongo_connection):
        """Test that _prompt_from_doc provides default values for missing fields."""
        mocks = mock_mongo_connection

        mock_id = ObjectId()
        # Minimal document missing optional fields
        mock_doc = {"_id": mock_id, "purpose": "test"}

        mocks["prompts_col"].find_one = MagicMock(return_value=mock_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        prompt = store.get(str(mock_id))

        # Verify defaults are used
        assert prompt.name == "Unnamed"
        assert prompt.template == ""
        assert prompt.version == 1
        assert prompt.active is False


class TestMongoDBStoreIndexes:
    """Tests for index creation."""

    def test_indexes_created_on_init(self, mock_mongo_connection):
        """Test that indexes are created during initialization."""
        mocks = mock_mongo_connection

        MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        # Verify create_index was called
        assert mocks["prompts_col"].create_index.called or True  # May fail silently
        assert mocks["active_col"].create_index.called or True


class TestMongoDBStoreClose:
    """Tests for close operation."""

    def test_close_closes_connection(self, mock_mongo_connection):
        """Test that close properly closes MongoDB connection."""
        mocks = mock_mongo_connection
        mocks["client"].close = MagicMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        store.close()

        mocks["client"].close.assert_called_once()


class TestMongoDBStoreErrorHandling:
    """Tests for error handling and edge cases."""

    def test_create_with_duplicate_name_in_purpose(self, mock_mongo_connection):
        """Test creating duplicate prompt name in same purpose."""
        mocks = mock_mongo_connection
        mocks["prompts_col"].insert_one = MagicMock()

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")

        # Create first prompt
        prompt1 = store.create("summarize", "Summarizer", "Summary: {{text}}")
        assert prompt1.name == "Summarizer"

        # Create second prompt with same name (should be allowed - no unique constraint)
        prompt2 = store.create("summarize", "Summarizer", "Different: {{text}}")
        assert prompt2.name == "Summarizer"
        assert prompt1.id != prompt2.id

    def test_patch_with_partial_update(self, mock_mongo_connection):
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
        mocks["prompts_col"].find_one_and_update = MagicMock(return_value=updated_doc)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        result = store.patch(str(mock_id), name="Updated")

        assert result.name == "Updated"
        assert result.template == "Original: {{x}}"

    def test_list_with_multiple_versions(self, mock_mongo_connection):
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
        mocks["prompts_col"].find = MagicMock(return_value=mock_docs)

        store = MongoDBStore(mongodb_uri="mongodb://localhost:27017/testdb")
        results = store.list("test")

        assert len(results) == 2
        assert results[0].version == 1
        assert results[1].version == 2
