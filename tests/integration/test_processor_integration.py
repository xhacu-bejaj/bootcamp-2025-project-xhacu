
import pytest
from app.services.processor import process_document
from app.models.domain import Prompt
from app.core.exceptions import PromptNotFoundError

@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_document_success(real_mongodb_store):
    """
    Test successful document processing using a real MongoDB store and a mock LLM.
    """
    store = real_mongodb_store
    user_id = "test_user_123"
    purpose = "test_purpose"

    # 1. Create a prompt
    prompt = await store.create(
        purpose=purpose,
        name="Test Prompt",
        template="Translate the following document: {{document_text}}",
    )

    # 2. Set the prompt as active for the user
    await store.set_active(user_id, purpose, prompt.id)


    # 3. Call the processor
    document_text = "Hello, world!"
    provider = "mock"
    response = await process_document(store, user_id, purpose, document_text, provider)

    # 4. Assert the response
    assert response is not None
    assert "[MOCK OUTPUT]" in response.output_text
    assert response.prompt_id == prompt.id
    assert response.prompt_version == prompt.version

@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_document_prompt_not_found(real_mongodb_store):
    """
    Test that PromptNotFoundError is raised when no active prompt is found.
    """
    store = real_mongodb_store
    user_id = "test_user_456"
    purpose = "non_existent_purpose"
    document_text = "This should fail."
    provider = "mock"

    with pytest.raises(PromptNotFoundError):
        await process_document(store, user_id, purpose, document_text, provider)
