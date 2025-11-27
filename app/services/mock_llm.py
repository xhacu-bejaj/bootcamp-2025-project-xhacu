from app.services.llm_client import LLMClient
from app.core import config

class MockLLM(LLMClient):
    def generate(self, prompt: str, **params):
        return {"text": f"[MOCK OUTPUT]\n{prompt[:200]} ...", "provider": "mock"}