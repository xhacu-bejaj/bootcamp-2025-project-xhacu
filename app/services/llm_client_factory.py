from app.services.llm_client import LLMClient
from app.services.google_llm import GoogleLLM
from app.services.mock_llm import MockLLM
from app.services.openai_llm import OpenaiLLM




PROVIDERS = {"mock": MockLLM, "openai": OpenaiLLM, "google": GoogleLLM}

class LLMClientFactory:
    @staticmethod
    def create_client(provider: str) -> LLMClient:
        provider= provider.lower()
        if provider == 'openai':
            return OpenaiLLM()
        
        elif provider == 'google':
            return GoogleLLM()
        
        elif provider == 'mock':
            return MockLLM()
        
        else:
            raise ValueError("Client not supported")
        
