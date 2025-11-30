from fastapi import HTTPException

from app.models.domain import Prompt
from app.models.schemas import PredictResponse
from .llm_client_factory import LLMClientFactory, Provider
from .prompt_store import InMemoryStore, PromptStore, UserId

def process_document(
        store: PromptStore, # our db where all the (active) prompts = purpose + template are stored
        user_id: UserId, # needed for prompt retrieval
        purpose: str, # needed for prompt retrieval (user_id, purpose) identifies a prompt
        document_text: str,
        provider: Provider ,
        **params,
    )-> PredictResponse | None:

    # Retrive the right active prompt
    active_prompt: Prompt | None = store.get_active(user_id, purpose)
    
    # Instanciate the client selected by the user
    try:
        llm_client = LLMClientFactory().create_client(provider)
    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'.")
    
    if active_prompt is not None:
        template = active_prompt.template 
    else:
        raise HTTPException(status_code=404, detail=f"No active prompt found for user: {user_id}, purpose: {purpose}")

    # API call to llm provider 
    post_process_doc = llm_client.generate(template, document_text, **params)
    return post_process_doc
    


