from app.models.domain import Prompt
from app.services.mock_llm import MockLLM


def test_mock_llm_generate_basic():
    llm = MockLLM()
    
    prompt = Prompt(
        id="p1",
        purpose="test",
        name="Test",
        template="Summarize: {{text}}",
        version=1
    )
    
    result = llm.generate(prompt, "Test document")
    
    assert result is not None
    assert result["provider"] == "mock"
    assert "text" in result
    assert "[MOCK OUTPUT]" in result["text"]
    assert "Summarize:" in result["text"]


def test_mock_llm_generate_includes_template():
    llm = MockLLM()
    
    prompt = Prompt(
        id="p2",
        purpose="extraction",
        name="Extractor",
        template="Extract key points from: {{content}}",
        version=2
    )
    
    result = llm.generate(prompt, "Important content")
    
    assert "Extract key points from:" in result["text"]
    assert result["provider"] == "mock"


def test_mock_llm_generate_with_kwargs():
    llm = MockLLM()
    
    prompt = Prompt(
        id="p3",
        purpose="translation",
        name="Translator",
        template="Translate to Spanish: {{text}}",
        version=1
    )
    
    result = llm.generate(
        prompt,
        "Hello world",
        temperature=0.9,
        max_tokens=100,
        top_p=0.8
    )
    
    assert result["provider"] == "mock"
    assert "[MOCK OUTPUT]" in result["text"]


def test_mock_llm_generate_empty_document():
    llm = MockLLM()
    
    prompt = Prompt(
        id="p4",
        purpose="analysis",
        name="Analyzer",
        template="Analyze: {{data}}",
        version=1
    )
    
    result = llm.generate(prompt, "")
    
    assert result is not None
    assert result["provider"] == "mock"
    assert "Analyze:" in result["text"]


def test_mock_llm_generate_multiple_calls():
    llm = MockLLM()
    
    prompt1 = Prompt(
        id="p5",
        purpose="purpose1",
        name="Prompt1",
        template="Template1: {{x}}",
        version=1
    )
    
    prompt2 = Prompt(
        id="p6",
        purpose="purpose2",
        name="Prompt2",
        template="Template2: {{y}}",
        version=1
    )
    
    result1 = llm.generate(prompt1, "doc1")
    result2 = llm.generate(prompt2, "doc2")
    
    assert result1["provider"] == "mock"
    assert result2["provider"] == "mock"
    assert "Template1:" in result1["text"]
    assert "Template2:" in result2["text"]
