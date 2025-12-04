import pytest
from unittest.mock import Mock, patch, MagicMock
from app.models.domain import Prompt
from app.services.google_llm import GoogleLLM
from app.models.schemas import PredictResponse, ModelInfo


@patch('app.services.google_llm.global_settings')
@patch('app.services.google_llm.genai.Client')
def test_google_llm_initialization(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = "test-api-key"
    
    llm = GoogleLLM()
    
    assert llm.model == "gemini-2.5-flash"
    assert llm.temperature == 0.5
    mock_client.assert_called_once_with(api_key="test-api-key")


@patch('app.services.google_llm.global_settings')
@patch('app.services.google_llm.genai.Client')
def test_google_llm_missing_api_key(mock_client, mock_settings):
    mock_settings.GOOGLE_API_KEY = None
    
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is missing or empty"):
        GoogleLLM()


@patch('app.services.google_llm.global_settings')
@patch('app.services.google_llm.genai.Client')
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
        version=1
    )
    
    result = llm.generate(prompt, "Test document")
    
    assert result is not None
    assert result.output_text == "This is a test response"
    assert result.model_info.model == "gemini-2.5-flash"
    assert result.prompt_id == "p1"


@patch('app.services.google_llm.global_settings')
@patch('app.services.google_llm.genai.Client')
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
        version=1
    )
    
    result = llm.generate(prompt, "Generate ideas", temperature=0.9)
    
    assert result is not None
    assert result.model_info.temperature == 0.9
    assert result.output_text == "High temperature response"


@patch('app.services.google_llm.global_settings')
@patch('app.services.google_llm.genai.Client')
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
        version=1
    )
    
    with pytest.raises(ValueError, match="LLM failed to return valid JSON"):
        llm.generate(prompt, "Test document")
