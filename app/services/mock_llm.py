from app.models.tables import Prompt
from app.services.llm_client import LLMClient
from app.core import config

class MockLLM(LLMClient):
    def generate(self, active_prompt: Prompt, document_text, **params):
        return {"text": f"[MOCK OUTPUT]\n{active_prompt.template} ...", "provider": "mock"}