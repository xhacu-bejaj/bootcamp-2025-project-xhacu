import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import os

from app.main import app
from app.services.mongodb_store import MongoDBStore
from app.core.config import settings

@pytest_asyncio.fixture(scope="function")
async def client():
    """
    An asynchronous test client for making API requests.
    This uses httpx.AsyncClient with an ASGITransport to correctly
    interface with the FastAPI app.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client

@pytest_asyncio.fixture(scope="function")
async def real_mongodb_store():
    """
    Provides a real, isolated MongoDB connection for a single test function.
    """
    if not settings.MONGODB_URI:
        pytest.skip("MONGODB_URI not set, skipping integration test.")
    
    original_db_name = settings.MONGODB_DB_NAME
    test_db_name = f"test_db_{os.urandom(8).hex()}"
    settings.MONGODB_DB_NAME = test_db_name
    
    store = MongoDBStore(mongodb_uri=settings.MONGODB_URI)
    try:
        await store.initialize()
        yield store
    finally:
        if settings.MONGODB_URI:
            try:
                await store.client.drop_database(test_db_name)
            except Exception as e:
                print(f"Error dropping test database {test_db_name}: {e}")
        
        await store.close()
        settings.MONGODB_DB_NAME = original_db_name