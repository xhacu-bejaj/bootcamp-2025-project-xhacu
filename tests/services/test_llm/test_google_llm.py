from fastapi import HTTPException
import pytest
from unittest.mock import Mock, patch
from app.models.domain import Prompt
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
    mock_settings.GOOGLE_API_KEY = None

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is missing or empty"):
        GoogleLLM()


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_generate_success(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "This is a test response", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.5}, "prompt_id": "p1", "prompt_version": 1, "latency_ms": 100}'

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()

    prompt = Prompt(
        id="p1",
        purpose="test",
        name="Test Prompt",
        template="Test template: {{text}}",
        version=1,
    )

    result = llm.generate(prompt, "Test document")

    assert result is not None
    assert result.output_text == "This is a test response"
    assert result.model_info.model == "gemini-2.5-flash"
    assert result.prompt_id == "p1"


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_generate_with_temperature_override(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "High temperature response", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.9}, "prompt_id": "p2", "prompt_version": 1, "latency_ms": 120}'

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()

    prompt = Prompt(
        id="p2",
        purpose="creative",
        name="Creative Prompt",
        template="Be creative: {{text}}",
        version=1,
    )

    result = llm.generate(prompt, "Generate ideas", temperature=0.9)

    assert result is not None
    assert result.model_info.temperature == 0.9
    assert result.output_text == "High temperature response"


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_generate_invalid_json_response(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = "Invalid JSON response"

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()

    prompt = Prompt(
        id="p3",
        purpose="test",
        name="Test Prompt",
        template="Test: {{text}}",
        version=1,
    )

    with pytest.raises(ValueError, match="LLM failed to return valid JSON"):
        llm.generate(prompt, "Test document")


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_config_key_filtering(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "filtered response", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.5}, "prompt_id": "p4", "prompt_version": 1, "latency_ms": 100}'

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()

    prompt = Prompt(
        id="p4", purpose="test", name="Test", template="Filter test: {{x}}", version=1
    )

    result = llm.generate(
        prompt, "doc", top_p=0.7, max_output_tokens=100, invalid_key="ignored"
    )

    assert result is not None
    assert result.output_text == "filtered response"
    call_args = mock_client_instance.models.generate_content.call_args
    assert "top_p" in str(call_args) or "max_output_tokens" in str(call_args)


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_handles_client_exceptions(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.side_effect = Exception("API error")
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    prompt = Prompt(
        id="p5", purpose="test", name="Test", template="Error test: {{x}}", version=1
    )

    with pytest.raises(HTTPException):
        llm.generate(prompt, "doc")


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_latency_is_int_and_nonnegative(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "test", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.5}, "prompt_id": "p6", "prompt_version": 1, "latency_ms": 50}'

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM()
    prompt = Prompt(
        id="p6",
        purpose="latency",
        name="Latency Test",
        template="Latency: {{x}}",
        version=1,
    )

    result = llm.generate(prompt, "doc")

    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


@patch("app.services.google_llm.settings")
@patch("app.services.google_llm.genai.Client")
def test_google_llm_temperature_default_and_override(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.text = '{"output_text": "temp test", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.3}, "prompt_id": "p7", "prompt_version": 1, "latency_ms": 75}'

    mock_client_instance = Mock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_client_instance

    llm = GoogleLLM(temperature=0.3)
    prompt = Prompt(
        id="p7", purpose="temp", name="Temp", template="Temp: {{x}}", version=1
    )

    result_default = llm.generate(prompt, "doc")
    assert result_default.model_info.temperature == 0.3

    mock_response.text = '{"output_text": "temp test", "model_info": {"model": "gemini-2.5-flash", "temperature": 0.8}, "prompt_id": "p7", "prompt_version": 1, "latency_ms": 75}'
    result_override = llm.generate(prompt, "doc", temperature=0.8)
    assert result_override.model_info.temperature == 0.8
