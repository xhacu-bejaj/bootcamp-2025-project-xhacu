import pytest
from app.models.domain import Prompt
from app.models.schemas import LLMOutput, LLMParams
from app.services.mock_llm import MockLLM


class TestMockLLM:
    """Tests for the MockLLM service."""

    @pytest.mark.asyncio
    async def test_mock_llm_generate_returns_llm_output(self):
        """Test that MockLLM.generate returns a valid LLMOutput object."""
        llm = MockLLM()
        doc = "Test document"
        prompt_template = "Summarize: {{ document_text }}"
        prompt_input = {"document_text": doc}
        
        # We are testing the LLM client, not the prompt rendering specifically.
        # So we pass the already rendered prompt string.
        rendered_prompt = prompt_template.replace("{{ document_text }}", doc)

        result = await llm.generate(rendered_prompt)

        assert isinstance(result, LLMOutput)
        assert result.output_text is not None
        assert "[MOCK OUTPUT]" in result.output_text
        assert "Summarize:" in result.output_text
        assert "Test document" in result.output_text # Ensure document text is included
        assert result.model_info.model == "mock"
        assert result.model_info.temperature == 0.5 # Default temperature

    @pytest.mark.asyncio
    async def test_mock_llm_with_custom_parameters_are_reflected(self):
        """Test MockLLM correctly reflects custom LLM parameters in its output."""
        llm = MockLLM()
        doc = "Hello world"
        prompt_template = "Translate to Spanish: {{ document_text }}"
        rendered_prompt = prompt_template.replace("{{ document_text }}", doc)

        custom_params = LLMParams(model="mock-custom-model", temperature=0.75)
        result = await llm.generate(rendered_prompt, params=custom_params)

        assert isinstance(result, LLMOutput)
        assert result.output_text is not None
        assert "[MOCK OUTPUT]" in result.output_text
        assert "Translate to Spanish:" in result.output_text
        assert "Hello world" in result.output_text # Ensure document text is included
        
        # Assert that custom parameters are reflected
        assert result.model_info.model == "mock-custom-model"
        assert result.model_info.temperature == 0.75
        assert isinstance(result.latency_ms, (int, float))

    @pytest.mark.asyncio
    async def test_mock_llm_empty_prompt(self):
        """Test MockLLM with an empty prompt string."""
        llm = MockLLM()
        result = await llm.generate("")

        assert isinstance(result, LLMOutput)
        assert result.output_text is not None
        assert "[MOCK OUTPUT]" in result.output_text
        assert result.model_info.model == "mock"

    @pytest.mark.asyncio
    async def test_mock_llm_response_structure(self):
        """Test that LLMOutput has all required fields and correct types."""
        llm = MockLLM()
        result = await llm.generate("Some prompt text")

        assert isinstance(result, LLMOutput)
        assert hasattr(result, "output_text")
        assert hasattr(result, "model_info")
        assert hasattr(result, "latency_ms")
        assert isinstance(result.output_text, str)
        assert isinstance(result.model_info, LLMParams)
        assert isinstance(result.latency_ms, (int, float))