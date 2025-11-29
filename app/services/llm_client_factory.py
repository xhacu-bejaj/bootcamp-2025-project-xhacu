from enum import Enum

from app.services.llm_client import LLMClient


class Provider(Enum):
    OPENAI = 'openai'
    GOOGLE = 'google'
    MOCK = 'mock'

class LLMClientFactory:
    @staticmethod
    def create_client(provider: Provider) -> LLMClient:
        if provider == Provider.OPENAI:
            from app.services.openai_llm import OpenaiLLM
            return OpenaiLLM()
        
        elif provider == Provider.GOOGLE:
            from app.services.google_llm import GoogleLLM
            return GoogleLLM()
        
        elif provider == Provider.MOCK:
            from app.services.mock_llm import MockLLM
            return MockLLM()
        
        else:
            raise ValueError("Client not supported")
        
#PROVIDERS = {"mock": MockLLM, "openai": OpenaiLLM, "google": GoogleLLM}