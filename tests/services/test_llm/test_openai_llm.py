import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from app.models.domain import Prompt
from app.services.openai_llm import OpenaiLLM


@patch('app.services.openai_llm.global_settings')
@patch('app.services.openai_llm.OpenAI')
def test_openai_llm_initialization(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"
    llm = OpenaiLLM()
    assert llm.model == 'gpt-4o-mini'
    assert llm.temperature == 0.5
    mock_openai.assert_called_once_with(api_key="openai-key")


@patch('app.services.openai_llm.global_settings')
@patch('app.services.openai_llm.OpenAI')
def test_openai_llm_missing_api_key(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = None
    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing or empty"):
        OpenaiLLM()


@patch('app.services.openai_llm.global_settings')
@patch('app.services.openai_llm.OpenAI')
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
    prompt = Prompt(id='p1', purpose='test', name='P', template='Do: {{text}}', version=1)

    res = llm.generate(prompt, "some doc")
    assert res is not None
    assert res.output_text == "OK from openai"
    assert res.model_info.model == llm.model


@patch('app.services.openai_llm.global_settings')
@patch('app.services.openai_llm.OpenAI')
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
    prompt = Prompt(id='p2', purpose='creative', name='P2', template='Create: {{text}}', version=1)

    res = llm.generate(prompt, "doc", temperature=0.9)
    assert res is not None
    assert res.model_info.temperature == 0.9
    assert res.output_text == "creative output"


@patch('app.services.openai_llm.global_settings')
@patch('app.services.openai_llm.OpenAI')
def test_openai_llm_api_error_raises_http_exception(mock_openai, mock_settings):
    mock_settings.OPENAI_API_KEY = "openai-key"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.side_effect = Exception("api down")
    mock_openai.return_value = mock_client_instance

    llm = OpenaiLLM()
    prompt = Prompt(id='p3', purpose='test', name='P3', template='Test: {{text}}', version=1)

    with pytest.raises(HTTPException, match="OpenAI API failed"):
        llm.generate(prompt, "doc")
