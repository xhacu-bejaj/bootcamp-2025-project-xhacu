from fastapi import HTTPException
import pytest
from app.services.processor import process_document
from app.services.prompt_store import InMemoryStore
from app.models.domain import Prompt
from app.models.schemas import LLMParams


def test_process_document_success():
    store = InMemoryStore()
    user_id = "user1"
    purpose = "summarization"
    
    prompt = store.create(purpose, "Summarizer", "Summarize the following: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    result = process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="This is a long document to summarize.",
        provider="mock"
    )
    
    assert result is not None
    assert "text" in result
    assert result["provider"] == "mock"


def test_process_document_no_active_prompt():
    store = InMemoryStore()
    user_id = "user2"
    purpose = "translation"
    
    with pytest.raises(ValueError, match="No active prompt found"):
        process_document(
            store=store,
            user_id=user_id,
            purpose=purpose,
            document_text="Text to translate",
            provider="mock"
        )


def test_process_document_invalid_provider():
    store = InMemoryStore()
    user_id = "user3"
    purpose = "analysis"
    
    prompt = store.create(purpose, "Analyzer", "Analyze: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    with pytest.raises(ValueError, match="Invalid provider"):
        process_document(
            store=store,
            user_id=user_id,
            purpose=purpose,
            document_text="Data to analyze",
            provider="invalid_provider"
        )


def test_process_document_with_llm_params():
    store = InMemoryStore()
    user_id = "user4"
    purpose = "extraction"
    
    prompt = store.create(purpose, "Extractor", "Extract: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    result = process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Important data to extract",
        provider="mock",
        temperature=0.5,
        max_tokens=100
    )
    
    assert result is not None
    assert result["provider"] == "mock"


def test_process_document_different_providers():
    store = InMemoryStore()
    user_id = "user5"
    purpose = "classification"
    
    prompt = store.create(purpose, "Classifier", "Classify: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    result_mock = process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Text to classify",
        provider="mock"
    )
    
    assert result_mock is not None
    assert result_mock["provider"] == "mock"


def test_process_document_multiple_purposes_same_user():
    store = InMemoryStore()
    user_id = "user6"
    
    p1 = store.create("summarization", "Summarizer", "Summarize: {{text}}")
    p2 = store.create("translation", "Translator", "Translate: {{text}}")
    
    store.set_active(user_id, "summarization", p1.id)
    store.set_active(user_id, "translation", p2.id)
    
    result1 = process_document(
        store=store,
        user_id=user_id,
        purpose="summarization",
        document_text="Document to summarize",
        provider="mock"
    )
    
    result2 = process_document(
        store=store,
        user_id=user_id,
        purpose="translation",
        document_text="Document to translate",
        provider="mock"
    )
    
    assert result1 is not None
    assert result2 is not None
    assert result1["provider"] == "mock"
    assert result2["provider"] == "mock"


def test_process_document_response_structure():
    store = InMemoryStore()
    user_id = "user7"
    purpose = "summarization"
    
    prompt = store.create(purpose, "Summarizer", "Summarize: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    result = process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Document for structure test",
        provider="mock"
    )
    
    assert "text" in result
    assert "provider" in result
    assert isinstance(result["text"], str)
    assert result["provider"] == "mock"


def test_process_document_preserves_prompt_metadata():
    store = InMemoryStore()
    user_id = "user8"
    purpose = "analysis"
    
    prompt = store.create(purpose, "Analyzer", "Analyze: {{text}}")
    store.set_active(user_id, purpose, prompt.id)
    
    result = process_document(
        store=store,
        user_id=user_id,
        purpose=purpose,
        document_text="Data to analyze",
        provider="mock"
    )
    
    assert result is not None
    assert "text" in result
    assert "Analyze:" in result["text"]

