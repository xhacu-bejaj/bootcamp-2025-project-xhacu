"""
Fixtures for end-to-end tests with full system integration.

These fixtures provide a complete running application for testing
full workflows from API to database.
"""
import os
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="function")
def e2e_client():
    """
    Provide a TestClient for end-to-end API testing.
    
    This uses the real application with all services configured.
    Uses function scope to ensure clean state between tests.
    """
    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def clean_chunk_store(e2e_client):
    """Clear chunk store before and after each test."""
    store = e2e_client.app.state.chunk_store
    await store.clear()
    yield store
    await store.clear()


@pytest.fixture
def sample_document_text():
    """Provide sample document text for testing."""
    return """
    Artificial Intelligence (AI) is revolutionizing the modern world.
    Machine learning, a subset of AI, enables computers to learn from data.
    Natural Language Processing (NLP) allows machines to understand human language.
    Deep learning uses neural networks to solve complex problems.
    """


@pytest.fixture
def sample_prompts():
    """Provide sample prompts for different purposes."""
    return {
        "summarize": {
            "name": "Document Summarizer",
            "template": "Please summarize the following document:\n\n{{ document_text }}"
        },
        "translate": {
            "name": "English Translator",
            "template": "Translate the following text to English:\n\n{{ document_text }}"
        },
        "extract": {
            "name": "Entity Extractor",
            "template": "Extract all key entities from:\n\n{{ document_text }}"
        }
    }
