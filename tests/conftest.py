import sys
import os

# MUST be set before any app imports to avoid MongoDB connection timeout
os.environ["MONGODB_URI"] = ""
os.environ["MONGODB_CONNECTION_STRING"] = ""

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_translate_prompt():
    """Create a default translate prompt for tests that need it."""
    from app.main import store
    
    prompt = store.create(
        purpose="translate",
        name="Default Translate",
        template="Translate the following document to English:\n\n{{ document_text }}"
    )
    store.set_active(user_id="user_anon", purpose="translate", prompt_id=prompt.id)
    yield
