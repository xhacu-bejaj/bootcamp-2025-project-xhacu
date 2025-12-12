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
    # Use context manager to ensure lifespan is triggered
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def setup_translate_prompt(client):
    """Create a default translate prompt for tests that need it."""
    # Access store from app state (set during lifespan)
    # Use client context to run async code synchronously
    import asyncio
    store = client.app.state.prompt_store
    
    # Run async methods in sync context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    prompt = loop.run_until_complete(store.create(
        purpose="translate",
        name="Default Translate",
        template="Translate the following document to English:\n\n{{ document_text }}"
    ))
    loop.run_until_complete(store.set_active(user_id="user_anon", purpose="translate", prompt_id=prompt.id))
    yield
