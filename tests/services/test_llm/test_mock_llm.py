import pytest
from app.models.domain import Prompt
from app.models.schemas import LLMOutput
from app.services.mock_llm import MockLLM


@pytest.mark.asyncio
async def test_mock_llm_generate_returns_predict_response():
    """Test that MockLLM.generate returns LLMOutput object"""
    llm = MockLLM()
    doc = "Test document"
    prompt = Prompt(
        id="p1", purpose="test", name="Test", template="Summarize: {{text}}", version=1
    )
    rendered_prompt = prompt.render({"text": doc})

    result = await llm.generate(rendered_prompt)

    assert isinstance(result, LLMOutput)
    assert result.output_text is not None
    assert "[MOCK OUTPUT]" in result.output_text
    assert "Summarize:" in result.output_text
    assert result.model_info.model == "mock"


@pytest.mark.asyncio
async def test_mock_llm_includes_document():
    """Test that MockLLM includes document in output"""
    llm = MockLLM()
    doc = "Important content"
    prompt = Prompt(
        id="p2",
        purpose="extraction",
        name="Extractor",
        template="Extract key points from: {{content}}",
        version=2,
    )
    rendered_prompt = prompt.render({"content": doc})
    result = await llm.generate(rendered_prompt)

    assert isinstance(result, LLMOutput)
    assert "Extract key points from:" in result.output_text
    assert "Important content" in result.output_text
    assert result.model_info.model == "mock"


@pytest.mark.asyncio
async def test_mock_llm_with_custom_parameters():
    """Test MockLLM with custom LLM parameters"""
    llm = MockLLM()
    doc = "Hello world"
    prompt = Prompt(
        id="p3",
        purpose="translation",
        name="Translator",
        template="Translate to Spanish: {{text}}",
        version=1,
    )
    rendered_prompt = prompt.render({"text": doc})

    # Custom params are accepted but ignored by MockLLM
    result = await llm.generate(rendered_prompt)

    assert isinstance(result, LLMOutput)
    assert "[MOCK OUTPUT]" in result.output_text
    assert "Translate to Spanish:" in result.output_text
    assert isinstance(result.latency_ms, (int, float))


@pytest.mark.asyncio
async def test_mock_llm_empty_document():
    """Test MockLLM with empty document"""
    llm = MockLLM()
    doc = ""
    prompt = Prompt(
        id="p4",
        purpose="analysis",
        name="Analyzer",
        template="Analyze: {{data}}",
        version=1,
    )
    rendered_prompt = prompt.render({"data": doc})
    result = await llm.generate(rendered_prompt)

    assert isinstance(result, LLMOutput)
    assert result.output_text is not None
    assert "Analyze:" in result.output_text
    assert result.model_info.model == "mock"


@pytest.mark.asyncio
async def test_mock_llm_multiple_calls():
    """Test MockLLM with multiple sequential calls"""
    llm = MockLLM()

    prompt1 = Prompt(
        id="p5",
        purpose="purpose1",
        name="Prompt1",
        template="Template1: {{x}}",
        version=1,
    )

    prompt2 = Prompt(
        id="p6",
        purpose="purpose2",
        name="Prompt2",
        template="Template2: {{y}}",
        version=1,
    )
    doc1 = "doc1"
    doc2 = "doc2"
    rendered1 = prompt1.render({"x": doc1})
    rendered2 = prompt2.render({"y": doc2})

    result1 = await llm.generate(rendered1)
    result2 = await llm.generate(rendered2)

    assert isinstance(result1, LLMOutput)
    assert isinstance(result2, LLMOutput)
    assert "Template1:" in result1.output_text
    assert "Template2:" in result2.output_text
    assert "doc1" in result1.output_text
    assert "doc2" in result2.output_text


@pytest.mark.asyncio
async def test_mock_llm_response_structure():
    """Test that LLMOutput has all required fields"""
    llm = MockLLM()
    doc = "test data"
    prompt = Prompt(
        id="p7", purpose="test", name="Test", template="Test: {{data}}", version=1
    )
    rendered_prompt = prompt.render({"data": doc})
    result = await llm.generate(rendered_prompt)

    assert isinstance(result, LLMOutput)
    assert hasattr(result, "output_text")
    assert hasattr(result, "model_info")
    assert hasattr(result, "latency_ms")
    assert result.model_info.model == "mock"
    assert isinstance(result.latency_ms, (int, float))
