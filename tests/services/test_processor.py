import pytest
from app.services.processor import process_document
from app.services.prompt_store import InMemoryStore
from app.models.schemas import PredictResponse, LLMParams


@pytest.mark.asyncio
async def test_process_document_returns_predict_response():
    """Test that process_document returns PredictResponse"""
    store = InMemoryStore()
    user_id = "user1"
    purpose = "summarization"

    prompt = await store.create(purpose, "Summarizer", "Summarize the following: {{text}}")
    await store.set_active(user_id, purpose, prompt.id)

    result = await process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="This is a long document to summarize.",
        provider="mock",
    )

    assert isinstance(result, PredictResponse)
    assert result.output_text is not None
    assert result.model_info.model == "mock"


@pytest.mark.asyncio
async def test_process_document_no_active_prompt():
    """Test that process_document raises error when no active prompt"""
    from app.core.exceptions import PromptNotFoundError
    
    store = InMemoryStore()
    user_id = "user2"
    purpose = "translation"

    with pytest.raises(PromptNotFoundError, match="No active prompt found"):
        await process_document(
            store=store,
            user_id=user_id,
            purpose=purpose,
            document_text="Text to translate",
            provider="mock",
        )


@pytest.mark.asyncio
async def test_process_document_with_custom_parameters():
    """Test process_document with custom LLM parameters"""
    store = InMemoryStore()
    user_id = "user4"
    purpose = "extraction"

    prompt = await store.create(purpose, "Extractor", "Extract: {{text}}")
    await store.set_active(user_id, purpose, prompt.id)

    params = LLMParams(model="mock", temperature=0.5)
    
    result = await process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Important data to extract",
        provider="mock",
        params=params,
    )

    assert isinstance(result, PredictResponse)
    assert result.model_info.model == "mock"
    assert result.prompt_id == prompt.id


@pytest.mark.asyncio
async def test_process_document_with_mock_provider():
    """Test process_document with mock provider"""
    store = InMemoryStore()
    user_id = "user5"
    purpose = "classification"

    prompt = await store.create(purpose, "Classifier", "Classify: {{text}}")
    await store.set_active(user_id, purpose, prompt.id)

    result = await process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Text to classify",
        provider="mock",
    )

    assert isinstance(result, PredictResponse)
    assert result.model_info.model == "mock"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_process_document_multiple_purposes():
    """Test process_document with multiple purposes for same user"""
    store = InMemoryStore()
    user_id = "user6"

    p1 = await store.create("summarization", "Summarizer", "Summarize: {{text}}")
    p2 = await store.create("translation", "Translator", "Translate: {{text}}")

    await store.set_active(user_id, "summarization", p1.id)
    await store.set_active(user_id, "translation", p2.id)

    result1 = await process_document(
        store=store,
        user_id=user_id,
        purpose="summarization",
        document_text="Document to summarize",
        provider="mock",
    )

    result2 = await process_document(
        store=store,
        user_id=user_id,
        purpose="translation",
        document_text="Document to translate",
        provider="mock",
    )

    assert isinstance(result1, PredictResponse)
    assert isinstance(result2, PredictResponse)
    assert result1.model_info.model == "mock"
    assert result2.model_info.model == "mock"


@pytest.mark.asyncio
async def test_process_document_response_has_all_fields():
    """Test that process_document response has all required fields"""
    store = InMemoryStore()
    user_id = "user7"
    purpose = "summarization"

    prompt = await store.create(purpose, "Summarizer", "Summarize: {{text}}")
    await store.set_active(user_id, purpose, prompt.id)

    result = await process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Document for structure test",
        provider="mock",
    )

    assert isinstance(result, PredictResponse)
    assert hasattr(result, "output_text")
    assert hasattr(result, "model_info")
    assert hasattr(result, "prompt_id")
    assert hasattr(result, "prompt_version")
    assert hasattr(result, "latency_ms")
    assert result.prompt_id == prompt.id


@pytest.mark.asyncio
async def test_process_document_preserves_prompt_metadata():
    """Test that process_document preserves prompt metadata"""
    store = InMemoryStore()
    user_id = "user8"
    purpose = "analysis"

    prompt = await store.create(purpose, "Analyzer", "Analyze: {{text}}")
    await store.set_active(user_id, purpose, prompt.id)

    result = await process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Data to analyze",
        provider="mock",
    )

    assert isinstance(result, PredictResponse)
    assert result.prompt_id == prompt.id
    assert result.prompt_version == prompt.version
    assert "Analyze:" in result.output_text
