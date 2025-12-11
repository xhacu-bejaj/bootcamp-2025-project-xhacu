"""Document processing service.

This module provides the core document processing functionality that:
- Retrieves active prompts for a user and purpose
- Creates appropriate LLM clients based on provider
- Generates processed documents using the LLM
"""
from app.models.domain import Prompt
from app.models.schemas import PredictResponse
from .llm_client_factory import LLMClientFactory
from .prompt_store import PromptStore, UserId
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
) -> PredictResponse | None:
    """Process a document using an LLM based on the active prompt.

    Retrieves the active prompt for the given user and purpose, creates an LLM client
    for the specified provider, and generates a processed version of the document.

    Args:
        store: The prompt store containing active prompts
        user_id: The ID of the user making the request
        purpose: The purpose of the prompt (e.g., 'translate', 'summarize')
        document_text: The raw text of the document to process
        provider: The LLM provider to use ('mock', 'openai', or 'google')
        **params: Additional parameters to pass to the LLM client

    Returns:
        PredictResponse containing the processed document and metadata,
        or None if processing fails

    Raises:
        ValueError: If no active prompt is found for the user/purpose combination
        ValueError: If the provider is invalid or not supported
    """

    active_prompt: Prompt | None = store.get_active(user_id, purpose)

    if not active_prompt:
        raise ValueError(
            f"No active prompt found for user: {user_id}, purpose: {purpose}"
        )

    prompt = active_prompt.render({"document_text": document_text})

    try:
        llm_client: LLMClient = LLMClientFactory().create_client(provider)
    except ValueError as e:
        raise ValueError(
            f"{e}: Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'."
        )

    return llm_client.generate(
        prompt,
        prompt_id=active_prompt.id,
        prompt_version=active_prompt.version,
        document_text=document_text,
        **params,
    )
