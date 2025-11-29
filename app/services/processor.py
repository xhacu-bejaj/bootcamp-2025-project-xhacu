from fastapi import HTTPException

from app.models.domain import Prompt
from app.models.schemas import PredictResponse
from .llm_client_factory import LLMClientFactory, Provider
from .prompt_store import InMemoryStore, PromptStore, UserId

def process_document(
        store: PromptStore, # our db where all the (active) prompts = purpose + template are stored
        user_id: UserId,
        purpose: str,
        document_text: str,
        provider: Provider = Provider.GOOGLE,
        **params,
    )-> PredictResponse | None:
    # maybe unpack Prompt , i do not need all the attrbutes
    active_prompt: Prompt | None = store.get_active(user_id, purpose)
    try:
        llm_client = LLMClientFactory().create_client(provider)
    except ValueError as e:
         raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'.")
    
    template = active_prompt.template # type: ignore

    post_process_doc = llm_client.generate(template, document_text, **params)
    return post_process_doc
    


