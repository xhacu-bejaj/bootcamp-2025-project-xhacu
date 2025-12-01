from fastapi import HTTPException

from app.models.domain import Prompt
from app.models.schemas import PredictResponse, LLMParams
from .llm_client_factory import LLMClientFactory
from .prompt_store import InMemoryStore, PromptStore, UserId
from app.core.logging import setup_logging, log_api_call



setup_logging()

@log_api_call
def process_document(
        store: PromptStore, # our db where all the (active) prompts = purpose + template are stored
        user_id: UserId, # needed for prompt retrieval
        purpose: str, # needed for prompt retrieval (user_id, purpose) identifies a prompt
        document_text: str,
        provider: str,
        **params, # model params
    )-> PredictResponse | None:

    # Retrive the right active prompt
    active_prompt: Prompt | None = store.get_active(user_id, purpose)
    
    # Instanciate the client selected by the user
    try:
        llm_client = LLMClientFactory().create_client(provider)
    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'.")
    
    if active_prompt is not None: 
        if active_prompt.template is not None:
            
            post_process_doc = llm_client.generate(active_prompt, document_text, **params)
        else:
            raise HTTPException(status_code=404, detail=f"Template not set for active prompt: {active_prompt.id}")
    else:
        raise HTTPException(status_code=404, detail=f"No active prompt found for user: {user_id}, purpose: {purpose}") 
    
    return post_process_doc
    


