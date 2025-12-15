import pytest
from app.services.llm_client_factory import LLMClientFactory
from app.services.mock_llm import MockLLM
from app.services.google_llm import GoogleLLM


class TestLLMClientFactory:
    """Tests for the LLMClientFactory."""

    def test_factory_create_mock_client(self):
        """Test that the factory creates a MockLLM client."""
        client = LLMClientFactory.create_client("mock")
        assert client is not None
        assert isinstance(client, MockLLM)

    def test_factory_create_google_client(self):
        """
        Test that the factory creates a GoogleLLM client.
        Note: This test assumes the GoogleLLM client can be instantiated
        without an API key, which is checked later during an actual API call.
        """
        client = LLMClientFactory.create_client("google")
        assert client is not None
        assert isinstance(client, GoogleLLM)

    def test_factory_create_invalid_provider_raises_error(self):
        """Test that creating a client with an invalid provider raises a ValueError."""
        with pytest.raises(ValueError, match="Client not supported"):
            LLMClientFactory.create_client("invalid_provider")

    def test_factory_create_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        client_lower = LLMClientFactory.create_client("mock")
        client_upper = LLMClientFactory.create_client("MOCK")
        client_mixed = LLMClientFactory.create_client("mOcK")

        assert isinstance(client_lower, MockLLM)
        assert isinstance(client_upper, MockLLM)
        assert isinstance(client_mixed, MockLLM)