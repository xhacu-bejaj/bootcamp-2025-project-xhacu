"""Document processing service.

This module provides the core document processing functionality that:
- Retrieves active prompts for a user and purpose
- Creates appropriate LLM clients based on provider
- Generates processed documents using the LLM
"""
from typing import Optional
from app.models.domain import Prompt
from app.models.schemas import LLMOutput, LLMParams, PredictResponse
from .llm_client_factory import LLMClientFactory
from .prompt_store import PromptStore, UserId
from app.core.logging import setup_logging, log_api_call
from app.core.exceptions import PromptNotFoundError, ConfigurationError
from app.services.llm_client import LLMClient

setup_logging()

@log_api_call
async def process_document(
    store: PromptStore,
    user_id: UserId,
    purpose: str,
    document_text: str,
    provider: str,
    params: Optional[LLMParams] = None,
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
        PromptNotFoundError: If no active prompt is found for the user/purpose combination
        ConfigurationError: If the provider is invalid or not supported
    """

    active_prompt: Prompt | None = await store.get_active(user_id, purpose)

    if not active_prompt or not active_prompt.template:
        raise PromptNotFoundError(
            f"No active prompt with a valid template found for user: {user_id}, purpose: {purpose}"
        )

    prompt = active_prompt.render({"document_text": document_text})

    try:
        llm_client: LLMClient = LLMClientFactory().create_client(provider)
    except ValueError as e:
        raise ConfigurationError(
            f"{e}: Invalid provider: {provider}. Must be one of 'mock', 'openai', or 'google'."
        )

    generated_content: LLMOutput | None = await llm_client.generate(prompt, params=params)
    
    if generated_content is None:
        return None

    return PredictResponse(
        output_text=generated_content.output_text,
        model_info=generated_content.model_info,
        prompt_id=active_prompt.id,
        prompt_version=active_prompt.version,
        latency_ms=generated_content.latency_ms,
    )

