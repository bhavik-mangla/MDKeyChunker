"""Tests for LLM client and prompt templates."""

import pytest
import json
from mdkeychunker.llm.prompt_templates import PromptTemplates, EXAMPLE_RESPONSES
from mdkeychunker.llm.llm_client import LLMClient
from mdkeychunker.utils.config import Config


def test_summary_prompt_generation():
    """Test summary prompt template."""
    templates = PromptTemplates()
    
    prompt = templates.summary_generation(
        "This is test content.",
        "Test Section"
    )
    
    assert "This is test content." in prompt
    assert "Test Section" in prompt


def test_question_prompt_generation():
    """Test question generation prompt."""
    templates = PromptTemplates()
    
    prompt = templates.question_generation(
        "Machine learning is a subset of AI.",
        "Introduction",
        num_questions=3
    )
    
    assert "Machine learning" in prompt
    assert "JSON" in prompt or "json" in prompt


def test_key_matching_prompt():
    """Test key matching prompt."""
    templates = PromptTemplates()
    
    prompt = templates.key_matching(
        "The transformer architecture uses attention.",
        "Deep Learning",
        ["transformer", "attention", "neural network"],
        {"transformer": 0.8, "attention": 0.7, "neural network": 0.5}
    )
    
    assert "transformer" in prompt
    assert "0.8" in prompt or "0.80" in prompt


def test_example_response_formats():
    """Test that example responses have expected structure."""
    # Questions example
    questions_resp = EXAMPLE_RESPONSES["questions"]["output"]
    assert "questions" in questions_resp
    assert isinstance(questions_resp["questions"], list)
    
    # Key matching example
    key_resp = EXAMPLE_RESPONSES["key_matching"]["output"]
    assert "matched" in key_resp
    assert "rejected" in key_resp
    assert "confidence" in key_resp
    assert isinstance(key_resp["matched"], list)
    assert isinstance(key_resp["confidence"], dict)


def test_llm_client_initialization():
    """Test LLM client initialization."""
    config = Config(
        llm_provider="openai",
        llm_api_key="test_key",
        llm_call_budget_per_doc=0  # Don't actually call API
    )
    
    # Should initialize without error
    client = LLMClient(config)
    assert client.provider == "openai"


def test_llm_json_parsing():
    """Test JSON response parsing."""
    config = Config(llm_provider="openai", llm_api_key="test")
    client = LLMClient(config)
    
    # Test direct JSON
    response1 = '{"matched": ["key1"], "rejected": []}'
    parsed1 = client._parse_json_response(response1)
    assert parsed1 is not None
    assert "matched" in parsed1
    
    # Test JSON in markdown code block
    response2 = '''Here's the result:
```json
{"matched": ["key2"], "rejected": []}
```
'''
    parsed2 = client._parse_json_response(response2)
    assert parsed2 is not None
    assert "matched" in parsed2
    
    # Test JSON with extra text
    response3 = 'The answer is {"matched": ["key3"], "rejected": []} as shown.'
    parsed3 = client._parse_json_response(response3)
    assert parsed3 is not None


def test_fallback_summary():
    """Test fallback summary generation."""
    config = Config(llm_call_budget_per_doc=0)
    client = LLMClient(config)
    
    text = "This is the first sentence. This is the second sentence. This is the third."
    summary = client._fallback_summary(text)
    
    assert len(summary) > 0
    assert summary.endswith('.')
    assert len(summary) < len(text)
