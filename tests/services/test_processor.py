"""
Tests for the document processor service.
"""

import pytest
from app.services.processor import process_document
from app.services.prompt_store import InMemoryStore
from app.models.schemas import PredictResponse, LLMParams
from app.core.exceptions import PromptNotFoundError, ConfigurationError

@pytest.fixture
def store() -> InMemoryStore:
    """Provides a clean in-memory store for each test."""
    return InMemoryStore()

@pytest.mark.asyncio
async def test_process_document_success(store: InMemoryStore):
    """Test that process_document successfully processes a document and returns a PredictResponse."""
    prompt = await store.create("translate", "Translator", "Translate: {{ document_text }}")
    await store.set_active("user1", "translate", prompt.id)

    result = await process_document(
        store=store,
        user_id="user1",
        purpose="translate",
        document_text="Hello world",
        provider="mock",
    )

    assert isinstance(result, PredictResponse)
    assert result.prompt_id == prompt.id
    assert "Hello world" in result.output_text  # Verify document text was rendered
    assert result.model_info.model == "mock"

@pytest.mark.asyncio
async def test_process_document_raises_if_no_active_prompt(store: InMemoryStore):
    """Test that process_document raises PromptNotFoundError if no active prompt is set."""
    with pytest.raises(PromptNotFoundError):
        await process_document(
            store=store,
            user_id="user2",
            purpose="nonexistent",
            document_text="Some text",
            provider="mock",
        )

@pytest.mark.asyncio
async def test_process_document_raises_if_provider_is_invalid(store: InMemoryStore):
    """Test that process_document raises ConfigurationError for an invalid provider."""
    prompt = await store.create("test", "Test", "Test: {{ document_text }}")
    await store.set_active("user3", "test", prompt.id)

    with pytest.raises(ConfigurationError):
        await process_document(
            store=store,
            user_id="user3",
            purpose="test",
            document_text="Some text",
            provider="invalid_provider",
        )

@pytest.mark.asyncio
async def test_process_document_uses_custom_params(store: InMemoryStore):
    """Test that process_document correctly passes custom LLM parameters to the client."""
    prompt = await store.create("test", "Test", "Test: {{ document_text }}")
    await store.set_active("user4", "test", prompt.id)
    
    custom_params = LLMParams(model="custom-mock-model", temperature=0.99)

    result = await process_document(
        store=store,
        user_id="user4",
        purpose="test",
        document_text="Some text",
        provider="mock",
        params=custom_params,
    )

    assert result is not None
    assert result.model_info.model == "custom-mock-model"
    assert result.model_info.temperature == 0.99