"""
Fixtures for integration tests with real dependencies.

These fixtures provide real MongoDB and ChromaDB connections for testing.
Tests are slower but verify actual integration behavior.
"""
import os
import pytest
import pytest_asyncio
from app.services.mongodb_store import MongoDBStore
from app.services.chunk_store import ChunkStore
from app.core.config import settings # Import the settings object



@pytest_asyncio.fixture
async def real_chunk_store():
    """
    Provide a real ChromaDB connection for integration tests.
    
    Uses the configured ChromaDB instance that gets cleaned up after tests.
    """
    # Use the default ChunkStore configuration
    store = ChunkStore()
    
    try:
        yield store
    finally:
        # Cleanup: clear the collection after test
        try:
            await store.clear()
        except Exception:
            pass  # Best effort cleanup


@pytest.fixture(scope="session")
def check_mongodb_available():
    """Check if MongoDB is available before running integration tests."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        pytest.skip("MONGODB_URI not set, skipping MongoDB integration tests")
    return True
