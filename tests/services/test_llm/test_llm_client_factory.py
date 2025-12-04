import pytest
from app.services.llm_client_factory import LLMClientFactory
from app.services.mock_llm import MockLLM


def test_factory_create_mock_client():
    factory = LLMClientFactory()
    client = factory.create_client("mock")
    
    assert client is not None
    assert isinstance(client, MockLLM)


def test_factory_create_invalid_provider():
    factory = LLMClientFactory()
    
    with pytest.raises(ValueError):
        factory.create_client("invalid_provider")


def test_factory_create_mock_lowercase():
    factory = LLMClientFactory()
    client = factory.create_client("mock")
    
    assert isinstance(client, MockLLM)


def test_factory_multiple_clients():
    factory = LLMClientFactory()
    
    client1 = factory.create_client("mock")
    client2 = factory.create_client("mock")
    
    assert client1 is not None
    assert client2 is not None
    assert isinstance(client1, MockLLM)
    assert isinstance(client2, MockLLM)
