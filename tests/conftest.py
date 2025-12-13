import sys
import os
import asyncio
import pytest
import pytest_asyncio

from fastapi.testclient import TestClient
from app.main import app
from app.services.mongodb_store import MongoDBStore # Import MongoDBStore
from app.core.config import settings # Import settings


@pytest.fixture(scope="session")
def client():
    # Use context manager to ensure lifespan is triggered
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_translate_prompt(): # Removed client as a dependency
    """Create a default translate prompt for tests that need it."""
    
    # Temporarily set the test DB name for this setup too
    original_db_name = settings.MONGODB_DB_NAME
    settings.MONGODB_DB_NAME = original_db_name + "_test"

    # Directly instantiate MongoDBStore using the test URI
    store = MongoDBStore(mongodb_uri=settings.MONGODB_URI)
    try:
        await store.initialize() # Initialize the store

        # Check if the default prompt already exists to prevent duplicates
        existing_prompts = await store.list(purpose="translate")
        if not any(p.name == "Default Translate" for p in existing_prompts):
            # Create a default translate prompt
            prompt = await store.create(
                purpose="translate",
                name="Default Translate",
                template="Translate the following document to English:\n\n{{ document_text }}"
            )
            await store.set_active(user_id="user_anon", purpose="translate", prompt_id=prompt.id)
        
        yield
    finally:
        await store.close() # Ensure store is closed
        # Revert DB name
        settings.MONGODB_DB_NAME = original_db_name


@pytest_asyncio.fixture(scope="function")
async def real_mongodb_store():
    """
    Provide a real MongoDB connection for integration tests.
    
    Requires MONGODB_URI to be set in app.core.config (from env var or .env).
    Uses a test database and cleans up after tests.
    """
    mongodb_uri = settings.MONGODB_URI
    if not mongodb_uri:
        pytest.skip("MONGODB_URI not set in app settings, skipping integration test")
    
    original_db_name = settings.MONGODB_DB_NAME
    # Temporarily override the DB name for tests
    settings.MONGODB_DB_NAME = original_db_name + "_test" # Append _test to the DB name
    
    # Create and initialize the store within the async context
    # Use the original mongodb_uri, but the store will pick up the new MONGODB_DB_NAME
    store = MongoDBStore(mongodb_uri=mongodb_uri) 
    
    try:
        await store.initialize() # Await the async initialization
        yield store
    finally:
        # Cleanup: drop test collections and revert DB name
        try:
            # Drop the entire test database for a clean slate
            await store.client.drop_database(settings.MONGODB_DB_NAME)
            await store.close()
        except Exception as e:
            print(f"Error during MongoDB cleanup: {e}") # Log cleanup errors
        finally:
            settings.MONGODB_DB_NAME = original_db_name # Revert DB name

