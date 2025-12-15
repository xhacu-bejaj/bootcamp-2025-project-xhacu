"""
Integration tests for the MongoDBStore service.

These tests require a live MongoDB connection and will be skipped if MONGODB_URI is not set.
They use the `real_mongodb_store` fixture to ensure each test runs against
a clean, isolated database.
"""

import pytest
from app.services.mongodb_store import MongoDBStore
from app.core.exceptions import PromptNotFoundError
from app.models.schemas import PredictResponse, LLMParams

# The 'real_mongodb_store' fixture is defined in tests/conftest.py
# and provides a fresh, isolated database for each test function.

@pytest.mark.usefixtures("real_mongodb_store")
class TestMongoDBStoreIntegration:
    """Integration tests for the MongoDBStore service."""

    @pytest.mark.asyncio
    async def test_create_and_get_prompt(self, real_mongodb_store: MongoDBStore):
        """Test creating a prompt and then retrieving it."""
        store = real_mongodb_store
        
        # Create a prompt
        created_prompt = await store.create(
            purpose="test_purpose", name="Test Prompt", template="Hello {{ name }}"
        )
        
        assert created_prompt.name == "Test Prompt"
        assert created_prompt.version == 1
        
        # Get the same prompt by ID
        retrieved_prompt = await store.get(created_prompt.id)
        
        assert retrieved_prompt is not None
        assert retrieved_prompt.id == created_prompt.id
        assert retrieved_prompt.name == "Test Prompt"

    @pytest.mark.asyncio
    async def test_get_nonexistent_prompt_raises_error(self, real_mongodb_store: MongoDBStore):
        """Test that getting a non-existent prompt raises PromptNotFoundError."""
        with pytest.raises(PromptNotFoundError):
            await real_mongodb_store.get("nonexistent-id")

    @pytest.mark.asyncio
    async def test_list_prompts(self, real_mongodb_store: MongoDBStore):
        """Test listing prompts by purpose."""
        store = real_mongodb_store
        
        await store.create(purpose="list_test", name="Prompt 1", template="1")
        await store.create(purpose="list_test", name="Prompt 2", template="2")
        await store.create(purpose="other_purpose", name="Prompt 3", template="3")
        
        # List by specific purpose
        results = await store.list(purpose="list_test")
        assert len(results) == 2
        
        # List all prompts
        all_results = await store.list()
        assert len(all_results) == 3

    @pytest.mark.asyncio
    async def test_patch_prompt(self, real_mongodb_store: MongoDBStore):
        """Test patching a prompt's name, template, and version."""
        store = real_mongodb_store
        
        prompt = await store.create(purpose="patch_test", name="Original", template="Original")
        assert prompt.version == 1

        # Patch the name and check version increment
        patched_prompt = await store.patch(prompt.id, name="Updated Name")
        assert patched_prompt.name == "Updated Name"
        assert patched_prompt.version == 2
        
        # Patch the template and check version increment again
        final_prompt = await store.patch(prompt.id, template="Updated Template")
        assert final_prompt.template == "Updated Template"
        assert final_prompt.name == "Updated Name" # Name should persist
        assert final_prompt.version == 3

    @pytest.mark.asyncio
    async def test_set_and_get_active_prompt(self, real_mongodb_store: MongoDBStore):
        """Test setting a prompt as active and then retrieving it."""
        store = real_mongodb_store
        user_id = "test_user"
        purpose = "active_test"
        
        prompt1 = await store.create(purpose=purpose, name="Prompt 1", template="1")
        prompt2 = await store.create(purpose=purpose, name="Prompt 2", template="2")

        # No active prompt should be set initially
        assert await store.get_active(user_id, purpose) is None
        
        # Set prompt2 as active
        await store.set_active(user_id, purpose, prompt2.id)
        
        # Verify prompt2 is now active
        active_prompt = await store.get_active(user_id, purpose)
        assert active_prompt is not None
        assert active_prompt.id == prompt2.id

        # Change the active prompt to prompt1
        await store.set_active(user_id, purpose, prompt1.id)
        new_active_prompt = await store.get_active(user_id, purpose)
        assert new_active_prompt.id == prompt1.id
        
    @pytest.mark.asyncio
    async def test_store_and_get_history(self, real_mongodb_store: MongoDBStore):
        """Test storing a response and retrieving it from history."""
        store = real_mongodb_store
        
        # Create a prompt to associate with the response
        prompt = await store.create(purpose="history_test", name="History", template="Test")
        
        # Create a mock response to store
        response_to_store = PredictResponse(
            output_text="Test output",
            model_info=LLMParams(model="test-model", temperature=0.5),
            prompt_id=prompt.id,
            prompt_version=prompt.version,
            latency_ms=123
        )
        
        await store.store_response(response_to_store, "history_user", "history_test")
        
        # Retrieve history
        history = await store.get_history(limit=10, user_id="history_user", purpose="history_test")
        
        assert len(history) == 1
        assert history[0]["prompt_id"] == prompt.id
        assert history[0]["user_id"] == "history_user"
        assert history[0]["model"] == "test-model"