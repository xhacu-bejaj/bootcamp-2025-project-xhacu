import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.models.domain import Prompt
from app.models.schemas import LLMParams, LLMOutput
from app.services.google_llm import GoogleLLM


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_initialization(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    llm = GoogleLLM()

    assert llm.model == "gemini-2.5-flash"
    assert llm.temperature == 0.5
    mock_client.assert_called_once_with(api_key="test-api-key")


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_missing_api_key(mock_client, mock_settings):
    from app.core.exceptions import ConfigurationError
    mock_settings.GOOGLE_API_KEY = None

    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY is missing or empty"):
        GoogleLLM()


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_generate_success(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "This is a test response"}'

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "Test document"
    prompt = Prompt(
        id="p1",
        purpose="test",
        name="Test Prompt",
        template="Test template: {{text}}",
        version=1,
    )
    rendered_prompt = prompt.render({"text": doc})

    result = await llm.generate(rendered_prompt)

    assert result is not None
    assert result.output_text == "This is a test response"
    assert result.model_info.model == "gemini-2.5-flash"


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_generate_with_temperature_override(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "High temperature response"}'

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "Generate ideas"
    prompt = Prompt(
        id="p2",
        purpose="creative",
        name="Creative Prompt",
        template="Be creative: {{text}}",
        version=1,
    )
    rendered_prompt = prompt.render({"text": doc})

    params = LLMParams(model="gemini-2.5-flash", temperature=0.9)
    result = await llm.generate(rendered_prompt, params=params)

    assert result is not None
    assert result.model_info.temperature == 0.9
    assert result.output_text == "High temperature response"


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_generate_invalid_json_response(mock_client, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = "Invalid JSON response"

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "Test document"
    prompt = Prompt(
        id="p3",
        purpose="test",
        name="Test Prompt",
        template="Test: {{text}}",
        version=1,
    )
    rendered_prompt = prompt.render({"text": doc})

    with pytest.raises(LLMGenerationError, match="LLM failed to return valid JSON"):
        await llm.generate(rendered_prompt)


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_config_key_filtering(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "filtered response"}'

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "doc"
    prompt = Prompt(
        id="p4", purpose="test", name="Test", template="Filter test: {{x}}", version=1
    )
    rendered_prompt = prompt.render({"x": doc})

    result = await llm.generate(rendered_prompt)

    assert result is not None
    assert isinstance(result, LLMOutput)
    assert result.output_text == "filtered response"
    # Verify generate_content was called with proper config
    call_args = mock_client_instance.aio.models.generate_content.call_args
    assert call_args is not None


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_handles_client_exceptions(mock_client, mock_settings):
    from app.core.exceptions import LLMGenerationError
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(side_effect=Exception("API error"))
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "doc"
    prompt = Prompt(
        id="p5", purpose="test", name="Test", template="Error test: {{x}}", version=1
    )
    rendered_prompt = prompt.render({"x": doc})

    with pytest.raises(LLMGenerationError, match="Model failed to generate content"):
        await llm.generate(rendered_prompt)


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_latency_is_int_and_nonnegative(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "test"}'

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    doc = "doc"
    prompt = Prompt(
        id="p6",
        purpose="latency",
        name="Latency Test",
        template="Latency: {{x}}",
        version=1,
    )
    rendered_prompt = prompt.render({"x": doc})

    result = await llm.generate(rendered_prompt)

    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
@pytest.mark.asyncio
async def test_google_llm_temperature_default_and_override(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "temp test"}'

    mock_client_instance = Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM(temperature=0.3)
    doc = "doc"
    prompt = Prompt(
        id="p7", purpose="temp", name="Temp", template="Temp: {{x}}", version=1
    )
    rendered_prompt = prompt.render({"x": doc})

    result_default = await llm.generate(rendered_prompt)
    assert result_default.model_info.temperature == 0.3

    mock_response.text = '{"output_text": "temp test"}'
    params = LLMParams(model="gemini-2.5-flash", temperature=0.8)
    result_override = await llm.generate(rendered_prompt, params=params)
    assert result_override.model_info.temperature == 0.8
