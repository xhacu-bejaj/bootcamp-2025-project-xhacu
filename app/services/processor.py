from fastapi import HTTPException

from app.models.domain import Prompt
from app.models.schemas import PredictResponse, LLMParams
from .llm_client_factory import LLMClientFactory
from .prompt_store import InMemoryStore, PromptStore, UserId
from app.core.logging import setup_logging, log_api_call
from app.services.llm_client import LLMClient



setup_logging()

@log_api_call
def process_document(
        store: PromptStore,
        user_id: UserId,
        purpose: str, 
        document_text: str,
        provider: str,
        **params, 
    )-> PredictResponse | None:

    # Retrive the right active prompt
    active_prompt: Prompt | None = store.get_active(user_id, purpose)
    
    # Instanciate the client selected by the user
    try:
        llm_client: LLMClient = LLMClientFactory().create_client(provider)
    except ValueError as e:
         raise ValueError(f"{e}: Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'.")
    
    if active_prompt is not None: 
        post_process_doc = llm_client.generate(active_prompt, document_text, **params)
    else:
        raise ValueError(f"No active prompt found for user: {user_id}, purpose: {purpose}") 
    
    return post_process_doc
    


