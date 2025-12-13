
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.dependencies import get_store
from app.models.domain import Prompt

@pytest.mark.integration
class TestPredictAPIIntegration:

    @pytest.mark.asyncio
    async def test_predict_with_real_store(self, real_mongodb_store):
        """
        Test the /v1/predict endpoint with a real MongoDB store.
        This test ensures the endpoint correctly integrates with the prompt store
        and the document processing logic.
        """
        # Override the get_store dependency to use the real_mongodb_store fixture
        app.dependency_overrides[get_store] = lambda: real_mongodb_store
        
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # 1. Setup: Create a prompt and set it as active for a test user
                user_id = "integration_test_user_predict"
                purpose = "integration_test_purpose"
                
                prompt = await real_mongodb_store.create(
                    purpose=purpose,
                    name="Integration Test Predict Prompt",
                    template="Process this for the test: {{document_text}}"
                )
                await real_mongodb_store.set_active(user_id, purpose, prompt.id)
                
                # 2. Action: Call the /v1/predict endpoint
                payload = {
                    "purpose": purpose,
                    "document_text": "Some document content.",
                    "provider": "mock"
                }
                headers = {
                    "X-User-Id": user_id
                }
                
                response = await client.post("/v1/predict", json=payload, headers=headers)
                
                # 3. Assertions
                assert response.status_code == 200
                data = response.json()
                
                assert data["output_text"] is not None
                assert "[MOCK OUTPUT]" in data["output_text"]
                assert "Some document content" in data["output_text"]
                assert data["prompt_id"] == prompt.id
                assert data["prompt_version"] == prompt.version
                assert data["model_info"]["model"] == "mock"
            
        finally:
            # Clean up the dependency override
            del app.dependency_overrides[get_store]
