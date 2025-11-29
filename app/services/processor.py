#from .llm_client import PROVIDERS
from enum import Enum
import enum
from ssl import Purpose

from fastapi import HTTPException

from app.models.domain import Prompt
from app.models.schemas import PredictRequest, PredictResponse
from .llm_client_factory import LLMClientFactory, Provider
from .prompt_store import InMemoryStore, PromptStore, UserId

def process_document(
        store: PromptStore, # our db where all the (active) prompts = purpose + template are stored
        user_id: UserId,
        purpose: str,
        document_text: str,
        provider: Provider = Provider.MOCK,
        **params,
    )->PredictResponse | None:
    store = InMemoryStore()
    active_prompt: Prompt | None = store.get_active(user_id, purpose)
    try:
        llm_client = LLMClientFactory().create_client(provider)
    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'.")
    ### Client instanciated, the input we will send to the Client will be template+doc
    ### which template to use depends on the purpose the user gave us 
    


