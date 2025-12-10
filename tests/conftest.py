import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_translate_prompt(client):
    """Setup a translate prompt once for all tests"""
    prompt_payload = {
        "purpose": "translate",
        "name": "Translator",
        "template": "Translate this text: {{document_text}}",
    }
    response = client.post("/v1/prompts", json=prompt_payload)
    if response.status_code == 200:
        # Get the prompt ID from the list endpoint
        list_response = client.get("/v1/prompts/translate")
        if list_response.status_code == 200 and list_response.json():
            prompt_id = list_response.json()[0]["id"]
            # Activate for user_anon
            client.post(
                f"/v1/prompts/{prompt_id}/activate?purpose=translate",
                headers={"x-user-id": "user_anon"},
            )
