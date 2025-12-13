
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.dependencies import get_store

@pytest.mark.integration
class TestPromptsAPIIntegration:

    @pytest.fixture(autouse=True)
    def override_dependency(self, real_mongodb_store):
        """Fixture to override the get_store dependency for all tests in this class."""
        app.dependency_overrides[get_store] = lambda: real_mongodb_store
        yield
        del app.dependency_overrides[get_store]

    @pytest.mark.asyncio
    async def test_create_prompt(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {
                "purpose": "test_create",
                "name": "Test Create Prompt",
                "template": "Template for creation test."
            }
            response = await client.post("/v1/prompts", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["id"] is not None
            assert data["name"] == "Test Create Prompt"
            assert data["purpose"] == "test_create"
            assert data["template"] == "Template for creation test."
            assert data["version"] == 1
            assert data["active"] is False

    @pytest.mark.asyncio
    async def test_list_prompts(self, real_mongodb_store):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create a prompt to ensure there is something to list
            await real_mongodb_store.create("test_list", "List Test 1", "template 1")
            await real_mongodb_store.create("test_list", "List Test 2", "template 2")

            response = await client.get("/v1/prompts/test_list")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 2
            assert data[0]["purpose"] == "test_list"

    @pytest.mark.asyncio
    async def test_patch_prompt(self, real_mongodb_store):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            prompt = await real_mongodb_store.create("test_patch", "Patch Me", "Initial template.")
            
            patch_payload = {"name": "Patched Name"}
            response = await client.patch(f"/v1/prompts/{prompt.id}", json=patch_payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Patched Name"
            assert data["version"] == 2

    @pytest.mark.asyncio
    async def test_activate_and_get_active_prompt(self, real_mongodb_store):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            user_id = "test_user_activation"
            purpose = "test_activation"
            prompt = await real_mongodb_store.create(purpose, "Activatable", "template")

            # Activate the prompt
            # Note: the endpoint is POST /v1/prompts/{prompt_id}/activate, and it takes `purpose` as a query parameter
            response_activate = await client.post(f"/v1/prompts/{prompt.id}/activate?purpose={purpose}", headers={"X-User-Id": user_id})
            assert response_activate.status_code == 200
            active_prompt_data = response_activate.json()
            assert active_prompt_data["id"] == prompt.id
            assert active_prompt_data["active"] is True

            # Get the active prompt
            # Note: the endpoint is GET /v1/get_active/{purpose}, and it takes `user_id` as a query parameter
            response_get = await client.get(f"/v1/get_active/{purpose}?user_id={user_id}")
            assert response_get.status_code == 200
            get_active_data = response_get.json()
            assert get_active_data["id"] == prompt.id
