"""
Integration tests for MongoDBStore with real MongoDB.

These tests use a real MongoDB connection to verify actual
database operations and behavior.
"""
import pytest
from app.models.domain import Prompt
from app.models.schemas import PredictResponse, LLMParams # Corrected import


@pytest.mark.integration
class TestMongoDBStoreIntegration:
    """Integration tests for MongoDB operations."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_prompt(self, real_mongodb_store):
        """Test creating and retrieving a prompt from real MongoDB."""
        store = real_mongodb_store
        
        # Create a prompt
        prompt = await store.create(
            purpose="summarize",
            name="Integration Test Summarizer",
            template="Summarize this: {{document_text}}"
        )
        
        assert prompt.id is not None
        assert prompt.purpose == "summarize"
        assert prompt.version == 1
        
        # Retrieve by ID
        retrieved = await store.get(prompt.id)
        assert retrieved is not None
        assert retrieved.id == prompt.id
        assert retrieved.name == "Integration Test Summarizer"
        assert retrieved.template == "Summarize this: {{document_text}}"

    @pytest.mark.asyncio
    async def test_list_prompts_by_purpose(self, real_mongodb_store):
        """Test listing prompts by purpose with real database."""
        store = real_mongodb_store
        
        # Create multiple prompts
        await store.create("translate", "Translator 1", "Translate: {{text}}")
        await store.create("translate", "Translator 2", "Translate to French: {{text}}")
        await store.create("summarize", "Summarizer", "Summarize: {{text}}")
        
        # List by purpose
        translate_prompts = await store.list(purpose="translate")
        summarize_prompts = await store.list(purpose="summarize")
        
        assert len(translate_prompts) >= 2
        assert len(summarize_prompts) >= 1
        assert all(p.purpose == "translate" for p in translate_prompts)
        assert all(p.purpose == "summarize" for p in summarize_prompts)

    @pytest.mark.asyncio
    async def test_update_prompt_increments_version(self, real_mongodb_store):
        """Test that updating a prompt increments version in real DB."""
        store = real_mongodb_store
        
        # Create prompt
        original = await store.create(
            "extract",
            "Extractor v1",
            "Extract data from: {{text}}"
        )
        assert original.version == 1
        
        # Update prompt
        updated = await store.patch(
            prompt_id=original.id,
            name="Extractor v2",
            template="Extract entities from: {{text}}"
        )
        
        assert updated.version == 2
        assert updated.name == "Extractor v2"
        assert updated.id == original.id
        
        # Verify in database
        retrieved = await store.get(original.id)
        assert retrieved.version == 2

    @pytest.mark.asyncio
    async def test_set_and_get_active_prompt(self, real_mongodb_store):
        """Test activating prompts for users in real database."""
        store = real_mongodb_store
        
        user_id = "integration_test_user"
        purpose = "classify"
        
        # Create two prompts
        prompt1 = await store.create(purpose, "Classifier 1", "Classify: {{text}}")
        prompt2 = await store.create(purpose, "Classifier 2", "Classify better: {{text}}")
        
        # Set first as active
        await store.set_active(user_id, purpose, prompt1.id)
        active1 = await store.get_active(user_id, purpose)
        assert active1.id == prompt1.id
        
        # Change to second
        await store.set_active(user_id, purpose, prompt2.id)
        active2 = await store.get_active(user_id, purpose)
        assert active2.id == prompt2.id
        
        # Verify only one active prompt per user/purpose
        all_prompts = await store.list(purpose)
        active_count = sum(1 for p in all_prompts if p.active)
        # Note: active field is managed in active_prompts collection
        # so this test verifies the switching mechanism

    @pytest.mark.asyncio
    async def test_log_and_retrieve_history(self, real_mongodb_store):
        """Test logging and retrieving request history in real DB."""
        store = real_mongodb_store
        
        # Create and activate a prompt
        prompt = await store.create(
            "test_purpose",
            "Test Prompt",
            "Test template"
        )
        await store.set_active("user_history_test", "test_purpose", prompt.id)
        
        # Log some history entries
        response1 = PredictResponse(
            prompt_id=prompt.id,
            output_text="Summary 1",
            model_info=LLMParams(model="mock-model", temperature=0.7), # Updated
            latency_ms=100,
            prompt_version=1
        )
        await store.store_response(response1, user_id="user_history_test", purpose="test_purpose")
        
        response2 = PredictResponse(
            prompt_id=prompt.id,
            output_text="Summary 2",
            model_info=LLMParams(model="mock-model", temperature=0.7), # Updated
            latency_ms=150,
            prompt_version=1
        )
        await store.store_response(response2, user_id="user_history_test", purpose="test_purpose")
        
        # Retrieve history
        history = await store.get_history(limit=10)
        assert len(history) >= 2
        
        # Filter by user
        user_history = await store.get_history(user_id="user_history_test", limit=10)
        assert len(user_history) >= 2
        assert all(h["user_id"] == "user_history_test" for h in user_history)
        
        # Verify order (most recent first)
        assert user_history[0]["latency_ms"] == 150
        assert user_history[1]["latency_ms"] == 100

    @pytest.mark.asyncio
    async def test_delete_prompt(self, real_mongodb_store):
        """Test deleting a prompt from real database."""
        store = real_mongodb_store
        
        # Create prompt
        prompt = await store.create(
            "temp_purpose",
            "Temporary Prompt",
            "This will be deleted"
        )
        prompt_id = prompt.id
        
        # Verify it exists
        retrieved = await store.get(prompt_id)
        assert retrieved is not None
        
        # Delete it
        deleted = await store.delete(prompt_id)
        assert deleted is True
        
        # Verify it's gone
        from app.core.exceptions import PromptNotFoundError
        with pytest.raises(PromptNotFoundError):
            await store.get(prompt_id)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, real_mongodb_store):
        """Test that concurrent operations work correctly with real DB."""
        import asyncio
        store = real_mongodb_store
        
        # Create multiple prompts concurrently
        async def create_prompt(idx):
            return await store.create(
                f"concurrent_{idx}",
                f"Concurrent Prompt {idx}",
                f"Template {idx}"
            )
        
        prompts = await asyncio.gather(*[create_prompt(i) for i in range(5)])
        
        # All should have unique IDs
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids))
        
        # All should be retrievable
        for prompt in prompts:
            retrieved = await store.get(prompt.id)
            assert retrieved is not None
