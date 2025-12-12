import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.models.domain import Prompt
from app.models.schemas import LLMParams, LLMOutput
from app.services.openai_llm import OpenaiLLM


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
def test_openai_llm_initialization(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"
    llm = OpenaiLLM()
    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.5
    mock_openai.assert_called_once_with(api_key="openai-key")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
def test_openai_llm_missing_api_key(mock_openai, mock_settings):
    from app.core.exceptions import ConfigurationError
    mock_settings.OPENAI_API_KEY = None
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is missing or empty"):
        OpenaiLLM()


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_generate_success(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    # Build fake response object
    fake_choice = Mock()
    fake_choice.message = Mock()
    fake_choice.message.content = '{"output_text": "OK from openai"}'
    fake_response = Mock()
    fake_response.choices = [fake_choice]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=fake_response)
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "some doc"
    prompt = Prompt(
        id="p1", purpose="test", name="P", template="Do: {{text}}", version=1
    )
    rendered_prompt = prompt.render({"text": doc})

    res = await llm.generate(rendered_prompt)
    assert res is not None
    assert isinstance(res, LLMOutput)
    assert res.output_text == "OK from openai"
    assert res.model_info.model == llm.model


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_generate_with_temperature_override(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    fake_choice = Mock()
    fake_choice.message = Mock()
    fake_choice.message.content = '{"output_text": "creative output"}'
    fake_response = Mock()
    fake_response.choices = [fake_choice]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=fake_response)
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p2", purpose="creative", name="P2", template="Create: {{text}}", version=1
    )
    rendered_prompt = prompt.render({"text": doc})

    res = await llm.generate(
        rendered_prompt,
        params=LLMParams(model="gpt-4o-mini", temperature=0.9)
    )
    assert res is not None
    assert isinstance(res, LLMOutput)
    assert res.model_info.temperature == 0.9
    assert res.output_text == "creative output"


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_api_error_raises_http_exception(mock_openai, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.OPENAI_API_KEY = "openai-key"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(side_effect=Exception("api down"))
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p3", purpose="test", name="P3", template="Test: {{text}}", version=1
    )
    rendered_prompt = prompt.render({"text": doc})

    with pytest.raises(LLMGenerationError, match="OpenAI API failed"):
        await llm.generate(rendered_prompt)


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_system_prompt_composition(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"output_text": "composed"}'))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p4",
        purpose="compose",
        name="Compose Test",
        template="Compose: {{x}}",
        version=1,
    )
    rendered_prompt = prompt.render({"x": doc})

    result = await llm.generate(rendered_prompt)
    assert result is not None
    assert isinstance(result, LLMOutput)

    call_args = mock_client_instance.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    assert any("Compose: " in str(m) for m in messages)


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_handles_null_response(mock_openai, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=None))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p6", purpose="null", name="Null Test", template="Null: {{x}}", version=1
    )
    rendered_prompt = prompt.render({"x": doc})

    with pytest.raises(LLMGenerationError):
        await llm.generate(rendered_prompt)


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_invalid_json_response(mock_openai, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Not valid JSON"))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p7", purpose="invalid", name="Invalid", template="Invalid: {{x}}", version=1
    )
    rendered_prompt = prompt.render({"x": doc})

    with pytest.raises(LLMGenerationError):
        await llm.generate(rendered_prompt)


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.AsyncOpenAI")
@pytest.mark.asyncio
async def test_openai_llm_api_exception_handling(mock_openai, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create = AsyncMock(side_effect=Exception(
        "API connection failed"
    ))
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    doc = "doc"
    prompt = Prompt(
        id="p8",
        purpose="api_error",
        name="API Error",
        template="Error: {{x}}",
        version=1,
    )
    rendered_prompt = prompt.render({"x": doc})

    with pytest.raises(LLMGenerationError):
        await llm.generate(rendered_prompt)


