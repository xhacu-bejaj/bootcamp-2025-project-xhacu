"""
Integration tests for the prompts API using an async client.
"""

import pytest
from httpx import AsyncClient
from app.main import app
from app.api.dependencies import get_store
from app.services.mongodb_store import MongoDBStore

@pytest.mark.usefixtures("real_mongodb_store")
class TestPromptsAPI:
    """Tests for the /v1/prompts endpoints using a real database."""

    @pytest.mark.asyncio
    async def test_create_and_list_prompt(self, client: AsyncClient, real_mongodb_store: MongoDBStore):
        """Test creating a prompt and then listing it shows exactly one prompt."""
        app.dependency_overrides[get_store] = lambda: real_mongodb_store
        
        purpose = "test_purpose"
        payload = {"purpose": purpose, "name": "Test Prompt", "template": "Test"}
        
        create_response = await client.post("/v1/prompts", json=payload)
        assert create_response.status_code == 200
        
        list_response = await client.get(f"/v1/prompts/{purpose}")
        assert list_response.status_code == 200
        list_data = list_response.json()
        
        assert len(list_data) == 1
        assert list_data[0]["name"] == "Test Prompt"
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_patch_prompt(self, client: AsyncClient, real_mongodb_store: MongoDBStore):
        """Test that patching a prompt updates it and increments the version."""
        app.dependency_overrides[get_store] = lambda: real_mongodb_store
        
        create_payload = {"purpose": "patch_test", "name": "Original", "template": "Original"}
        create_response = await client.post("/v1/prompts", json=create_payload)
        prompt_id = create_response.json()["id"]

        patch_payload = {"name": "Updated Name"}
        patch_response = await client.patch(f"/v1/prompts/{prompt_id}", json=patch_payload)
        assert patch_response.status_code == 200
        patch_data = patch_response.json()
        
        assert patch_data["name"] == "Updated Name"
        assert patch_data["version"] == 2
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_activate_and_get_active_prompt(self, client: AsyncClient, real_mongodb_store: MongoDBStore):
        """Test activating a prompt and retrieving it."""
        app.dependency_overrides[get_store] = lambda: real_mongodb_store
        
        purpose = "activation_test"
        payload = {"purpose": purpose, "name": "Prompt 1", "template": "1"}
        prompt_id = (await client.post("/v1/prompts", json=payload)).json()["id"]
        user_id = "test_user_123"

        await client.post(f"/v1/prompts/{prompt_id}/activate?purpose={purpose}", headers={"x-user-id": user_id})
        
        get_active_response = await client.get(f"/v1/get_active/{purpose}?user_id={user_id}")
        assert get_active_response.status_code == 200
        assert get_active_response.json()["id"] == prompt_id
        
        app.dependency_overrides.clear()
