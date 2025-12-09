import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from app.models.domain import Prompt
from app.services.openai_llm import OpenaiLLM


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_initialization(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"
    llm = OpenaiLLM()
    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.5
    mock_openai.assert_called_once_with(api_key="openai-key")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_missing_api_key(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = None
    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing or empty"):
        OpenaiLLM()


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_generate_success(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    # Build fake response object
    fake_choice = Mock()
    fake_choice.message = Mock()
    fake_choice.message.content = '{"output_text": "OK from openai"}'
    fake_response = Mock()
    fake_response.choices = [fake_choice]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = fake_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p1", purpose="test", name="P", template="Do: {{text}}", version=1
    )

    res = llm.generate(prompt, "some doc")
    assert res is not None
    assert res.output_text == "OK from openai"
    assert res.model_info.model == llm.model


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_generate_with_temperature_override(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    fake_choice = Mock()
    fake_choice.message = Mock()
    fake_choice.message.content = '{"output_text": "creative output"}'
    fake_response = Mock()
    fake_response.choices = [fake_choice]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = fake_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p2", purpose="creative", name="P2", template="Create: {{text}}", version=1
    )

    res = llm.generate(prompt, "doc", temperature=0.9)
    assert res is not None
    assert res.model_info.temperature == 0.9
    assert res.output_text == "creative output"


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_api_error_raises_http_exception(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.side_effect = Exception("api down")
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p3", purpose="test", name="P3", template="Test: {{text}}", version=1
    )

    with pytest.raises(HTTPException, match="OpenAI API failed"):
        llm.generate(prompt, "doc")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_system_prompt_composition(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"output_text": "composed"}'))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p4",
        purpose="compose",
        name="Compose Test",
        template="Compose: {{x}}",
        version=1,
    )

    result = llm.generate(prompt, "doc")
    assert result is not None

    call_args = mock_client_instance.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    assert any("Compose: " in str(m) for m in messages)


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_api_params_forwarding(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"output_text": "forwarded"}'))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p5", purpose="forward", name="Forward", template="Forward: {{x}}", version=1
    )

    llm.generate(prompt, "doc", top_p=0.9, max_tokens=500, frequency_penalty=0.2)

    call_args = mock_client_instance.chat.completions.create.call_args
    kwargs = call_args.kwargs if hasattr(call_args, "kwargs") else call_args[1]
    assert kwargs.get("top_p") == 0.9
    assert kwargs.get("max_tokens") == 500
    assert kwargs.get("frequency_penalty") == 0.2


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_handles_null_response(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=None))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p6", purpose="null", name="Null Test", template="Null: {{x}}", version=1
    )

    with pytest.raises(ValueError):
        llm.generate(prompt, "doc")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_invalid_json_response(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Not valid JSON"))]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p7", purpose="invalid", name="Invalid", template="Invalid: {{x}}", version=1
    )

    with pytest.raises(ValueError):
        llm.generate(prompt, "doc")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_api_exception_handling(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.side_effect = Exception(
        "API connection failed"
    )
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p8",
        purpose="api_error",
        name="API Error",
        template="Error: {{x}}",
        version=1,
    )

    with pytest.raises(HTTPException):
        llm.generate(prompt, "doc")


@patch("app.services.openai_llm.settings")
@patch("app.services.openai_llm.OpenAI")
def test_openai_llm_response_format_override(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "test-api-key"

    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content='{"output_text": "format_test"}'))
    ]

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(
        id="p9", purpose="format", name="Format", template="Format: {{x}}", version=1
    )

    result = llm.generate(prompt, "doc", response_format={"type": "json_object"})
    assert result is not None

    call_args = mock_client_instance.chat.completions.create.call_args
    kwargs = call_args.kwargs if hasattr(call_args, "kwargs") else call_args[1]
    assert kwargs.get("response_format") == {"type": "json_object"}
