"""
Tests for the prediction/processing API endpoints.
"""

import pytest
from httpx import AsyncClient
from app.main import app
from app.api.dependencies import get_store
from unittest.mock import MagicMock, AsyncMock
from app.models.domain import Prompt


@pytest.fixture(autouse=True) # Moved fixture outside the class
def mock_get_store_dependency():
    """
    Overrides the get_store dependency to return a mock PromptStore.
    This mock provides a predictable prompt for tests.
    """
    mock_store = MagicMock()
    mock_prompt = Prompt(
        id="mock_prompt_id",
        purpose="translate",
        name="Default Translate",
        template="Translate the following document to English:\n\n{{ document_text }}",
        version=1,
        active=True
    )
    mock_store.get_active = AsyncMock(return_value=mock_prompt)
    
    # Override get_store to return our mock
    app.dependency_overrides[get_store] = lambda: mock_store
    yield mock_store
    app.dependency_overrides.clear()


class TestPredictAPI:
    """Test prediction/processing endpoints"""

    @pytest.mark.asyncio
    async def test_predict_with_mock_provider(self, client: AsyncClient):
        """Test POST /v1/predict with mock provider"""
        payload = {
            "purpose": "translate",
            "document_text": "Hello world",
            "provider": "mock",
        }
        response = await client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["output_text", "model_info", "prompt_id", "prompt_version", "latency_ms"]
        for field in required_fields:
            assert field in data
            
        assert data["model_info"]["model"] == "mock"
        assert "[MOCK OUTPUT]" in data["output_text"]
        assert "Hello world" in data["output_text"]

    @pytest.mark.asyncio
    async def test_predict_with_document_variations(self, client: AsyncClient):
        """Test predict with various documents"""
        documents = [
            "Short text",
            "Special characters: @#$%^&*()",
            "", # Empty document
        ]
        for doc in documents:
            payload = {"purpose": "translate", "document_text": doc, "provider": "mock"}
            response = await client.post("/v1/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert doc in data["output_text"]

    @pytest.mark.asyncio
    async def test_predict_required_fields_missing(self, client: AsyncClient):
        """Test predict with missing required fields returns 422"""
        # Missing document_text
        payload = {"purpose": "translate", "provider": "mock"}
        response = await client.post("/v1/predict", json=payload)
        assert response.status_code == 422

        # Missing purpose
        payload = {"document_text": "test", "provider": "mock"}
        response = await client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_with_custom_params(self, client: AsyncClient):
        """Test predict with custom LLM parameters"""
        payload = {
            "purpose": "translate",
            "document_text": "Test document",
            "provider": "mock",
            "params": {"model": "mock-custom", "temperature": 0.8},
        }
        response = await client.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model_info"]["model"] == "mock-custom"
        assert data["model_info"]["temperature"] == 0.8