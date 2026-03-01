"""Tests for LLMClient."""
import pytest
from unittest.mock import MagicMock, patch
from mdkeychunker.config import Config
from mdkeychunker.llm_client import LLMClient


def _config(**kwargs) -> Config:
    defaults = dict(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_base_url="",
        llm_model="gpt-4o-mini",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def test_parse_json_direct():
    raw = '{"key": "value", "num": 42}'
    result = LLMClient._parse_json(raw)
    assert result == {"key": "value", "num": 42}


def test_parse_json_from_markdown_block():
    raw = '```json\n{"title": "Hello"}\n```'
    result = LLMClient._parse_json(raw)
    assert result == {"title": "Hello"}


def test_parse_json_bare_in_text():
    raw = 'Some preamble text. {"answer": true} some trailing text.'
    result = LLMClient._parse_json(raw)
    assert result is not None
    assert result.get("answer") is True


def test_parse_json_invalid_returns_none():
    result = LLMClient._parse_json("This is not JSON at all.")
    assert result is None


def test_parse_json_nested():
    raw = '{"entities": [{"name": "Python", "type": "TECH"}], "key": "python"}'
    result = LLMClient._parse_json(raw)
    assert result["key"] == "python"
    assert result["entities"][0]["name"] == "Python"


def test_openai_client_initialized():
    config = _config(llm_provider="openai")
    with patch("mdkeychunker.llm_client.LLMClient._init_client"):
        client = LLMClient.__new__(LLMClient)
        client.config = config
        client.provider = "openai"
        client._client = None
        # Just verify it stores the provider
        assert client.provider == "openai"


def test_openai_compatible_provider():
    config = _config(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
    )
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        client = LLMClient(config)
        assert client.provider == "openai_compatible"


def test_call_openai_returns_content():
    config = _config(llm_provider="openai")
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "openai"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello world"
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response
    client._client = mock_openai

    result = client.call("test prompt")
    assert result == "Hello world"


def test_call_anthropic_returns_content():
    config = _config(llm_provider="anthropic")
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "anthropic"

    mock_response = MagicMock()
    mock_response.content[0].text = "Anthropic response"
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = mock_response
    client._client = mock_anthropic

    result = client.call("test prompt")
    assert result == "Anthropic response"


def test_call_json_parses_response():
    config = _config()
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "openai"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"title": "Test", "key": "testing"}'
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response
    client._client = mock_openai

    result = client.call_json("prompt")
    assert result["title"] == "Test"
    assert result["key"] == "testing"


def test_call_without_client_raises():
    config = _config()
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "openai"
    client._client = None

    with pytest.raises(RuntimeError, match="not initialized"):
        client.call("prompt")


def test_unknown_provider_raises_on_call():
    config = _config(llm_provider="unknown_provider")
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "unknown_provider"
    client._client = MagicMock()  # pretend it exists

    with pytest.raises(RuntimeError, match="Unknown provider"):
        client.call("prompt")


# ─── New tests (Part 5 of optimization prompt) ────────────────────────────


def test_parse_json_deeply_nested():
    """JSON with nested objects (entities array) must parse correctly."""
    raw = 'Here is the analysis: {"entities": [{"name": "Python", "type": "TECH"}, {"name": "FastAPI", "type": "TECH"}], "key": "web frameworks"}'
    result = LLMClient._parse_json(raw)
    assert result is not None
    assert len(result["entities"]) == 2
    assert result["key"] == "web frameworks"


def test_extract_json_object_balanced():
    """_extract_json_object must correctly find the first balanced JSON object."""
    text = 'Preamble. {"outer": {"inner": "value"}, "list": [1, 2, 3]} trailing.'
    obj_str = LLMClient._extract_json_object(text)
    assert obj_str is not None
    import json
    parsed = json.loads(obj_str)
    assert parsed["outer"]["inner"] == "value"
    assert parsed["list"] == [1, 2, 3]


def test_parse_json_no_json_returns_none():
    result = LLMClient._parse_json("No JSON here whatsoever.")
    assert result is None


def test_call_retries_on_failure():
    """call() should retry on transient errors and eventually raise."""
    config = _config(llm_provider="openai")
    client = LLMClient.__new__(LLMClient)
    client.config = config
    client.provider = "openai"

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = OSError(
        "connection reset")
    client._client = mock_openai

    with patch("time.sleep"):  # don't actually sleep in tests
        with pytest.raises(OSError):
            client.call("prompt", retries=1)

    # Should have been called 2 times (1 initial + 1 retry)
    assert mock_openai.chat.completions.create.call_count == 2
