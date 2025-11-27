#from .llm_client import PROVIDERS
from .llm_client_factory import Provider
from .prompt_store import PromptStore

def process_document(
        store: PromptStore,
        user_id: str,
        purpose: str,
        document_text: str,
        provider: str = "mock",
        **params,
    ):
    ...
