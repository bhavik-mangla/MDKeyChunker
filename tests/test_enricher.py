"""Tests for Enricher."""
import pytest
from unittest.mock import MagicMock
from mdkeychunker.enricher import Enricher
from mdkeychunker.models import Chunk


def _make_chunk(text="Sample text about machine learning.", section="Introduction"):
    return Chunk(text=text, section_title=section)


def _mock_llm(result: dict):
    llm = MagicMock()
    llm.call_json.return_value = result
    return llm


SAMPLE_RESULT = {
    "title": "Machine Learning Overview",
    "summary": "This chunk discusses ML basics.",
    "keywords": ["machine learning", "supervised", "model"],
    "entities": [{"name": "scikit-learn", "type": "TECH"}],
    "questions": ["What is machine learning?"],
    "key": "machine learning",
    "related_keys": [],
}


def test_enrich_single_chunk():
    llm = _mock_llm(SAMPLE_RESULT)
    enricher = Enricher(llm)
    chunks = [_make_chunk()]
    result = enricher.enrich_chunks(chunks)

    assert result[0].title == "Machine Learning Overview"
    assert result[0].summary == "This chunk discusses ML basics."
    assert result[0].key == "machine learning"
    assert result[0].keywords == ["machine learning", "supervised", "model"]
    assert result[0].entities == [{"name": "scikit-learn", "type": "TECH"}]
    assert result[0].questions == ["What is machine learning?"]


def test_rolling_keys_accumulate():
    llm = _mock_llm(SAMPLE_RESULT)
    enricher = Enricher(llm)

    chunks = [_make_chunk() for _ in range(3)]
    enricher.enrich_chunks(chunks)

    # After processing, rolling_keys should contain the key
    assert "machine learning" in enricher.rolling_keys
    assert enricher.rolling_keys["machine learning"]["count"] == 3


def test_rolling_keys_passed_to_prompt():
    llm = MagicMock()
    llm.call_json.return_value = {**SAMPLE_RESULT, "key": "deep learning"}

    enricher = Enricher(llm)
    chunks = [_make_chunk("Chunk 1"), _make_chunk("Chunk 2")]
    enricher.enrich_chunks(chunks)

    # The second call should have rolling keys in the prompt
    second_call_args = llm.call_json.call_args_list[1]
    prompt_text = second_call_args[0][0]
    assert "deep learning" in prompt_text


def test_llm_failure_leaves_chunk_intact():
    llm = _mock_llm(None)  # LLM returns None
    enricher = Enricher(llm)
    chunk = _make_chunk("Some text")
    result = enricher.enrich_chunks([chunk])

    # Chunk should be returned unchanged
    assert result[0].text == "Some text"
    assert result[0].title == ""
    assert result[0].key == ""


def test_related_keys_populated():
    first_result = {**SAMPLE_RESULT,
                    "key": "neural networks", "related_keys": []}
    second_result = {
        **SAMPLE_RESULT,
        "key": "deep learning",
        "related_keys": ["neural networks"],
    }

    llm = MagicMock()
    llm.call_json.side_effect = [first_result, second_result]

    enricher = Enricher(llm)
    chunks = [_make_chunk("Chunk A"), _make_chunk("Chunk B")]
    result = enricher.enrich_chunks(chunks)

    assert result[1].related_keys == ["neural networks"]


def test_first_chunk_prompt_has_no_rolling_keys():
    llm = MagicMock()
    llm.call_json.return_value = SAMPLE_RESULT
    enricher = Enricher(llm)
    enricher.enrich_chunks([_make_chunk()])

    prompt = llm.call_json.call_args_list[0][0][0]
    assert "none yet" in prompt.lower() or "(none" in prompt


def test_rolling_keys_track_first_and_last_chunk():
    llm = MagicMock()
    llm.call_json.side_effect = [
        {**SAMPLE_RESULT, "key": "topic A"},
        {**SAMPLE_RESULT, "key": "topic B"},
        {**SAMPLE_RESULT, "key": "topic A"},
    ]
    enricher = Enricher(llm)
    enricher.enrich_chunks([_make_chunk() for _ in range(3)])

    assert enricher.rolling_keys["topic A"]["first_chunk"] == 0
    assert enricher.rolling_keys["topic A"]["last_chunk"] == 2
    assert enricher.rolling_keys["topic A"]["count"] == 2


def test_enrich_empty_list():
    llm = _mock_llm(SAMPLE_RESULT)
    enricher = Enricher(llm)
    result = enricher.enrich_chunks([])
    assert result == []


# ─── New tests (Part 5 of optimization prompt) ────────────────────────────


def test_reset_clears_rolling_keys():
    """reset() must clear rolling keys so documents don't leak into each other."""
    llm = _mock_llm(SAMPLE_RESULT)
    enricher = Enricher(llm)
    enricher.enrich_chunks([_make_chunk()])
    assert len(enricher.rolling_keys) > 0
    enricher.reset()
    assert len(enricher.rolling_keys) == 0


def test_rolling_keys_pruned_at_limit():
    """Rolling keys must not exceed MAX_ROLLING_KEYS (40)."""
    from mdkeychunker.enricher import MAX_ROLLING_KEYS
    llm = MagicMock()
    results = [
        {**SAMPLE_RESULT, "key": f"unique topic {i}", "related_keys": []}
        for i in range(60)
    ]
    llm.call_json.side_effect = results
    enricher = Enricher(llm)
    enricher.enrich_chunks([_make_chunk() for _ in range(60)])
    assert len(enricher.rolling_keys) <= MAX_ROLLING_KEYS


def test_format_rolling_keys_shows_counts():
    """_format_rolling_keys should list each key with its seen count."""
    llm = MagicMock()
    llm.call_json.side_effect = [
        {**SAMPLE_RESULT, "key": "topic A"},
        {**SAMPLE_RESULT, "key": "topic B"},
        {**SAMPLE_RESULT, "key": "topic A"},
    ]
    enricher = Enricher(llm)
    enricher.enrich_chunks([_make_chunk() for _ in range(3)])
    formatted = enricher._format_rolling_keys()
    assert "topic A" in formatted
    assert "2x" in formatted  # topic A seen twice
    assert "topic B" in formatted


def test_llm_exception_does_not_crash_pipeline():
    """If LLM throws an exception, the chunk should be returned unenriched."""
    llm = MagicMock()
    llm.call_json.side_effect = ConnectionError("Network error")
    enricher = Enricher(llm)
    chunk = _make_chunk("Some text")
    result = enricher.enrich_chunks([chunk])
    assert result[0].text == "Some text"
    assert result[0].key == ""  # unenriched


def test_prompt_includes_position_and_prev_summary():
    """Prompt must include chunk position and previous chunk summary."""
    llm = MagicMock()
    llm.call_json.return_value = SAMPLE_RESULT
    enricher = Enricher(llm)
    chunks = [_make_chunk("First"), _make_chunk("Second")]
    chunks[0].summary = ""  # first chunk has no prior summary
    enricher.enrich_chunks(chunks)

    # First chunk prompt should say position 1 of 2
    first_prompt = llm.call_json.call_args_list[0][0][0]
    assert "1 of 2" in first_prompt
    assert "first chunk" in first_prompt.lower()

    # Second chunk prompt should mention position 2 of 2
    second_prompt = llm.call_json.call_args_list[1][0][0]
    assert "2 of 2" in second_prompt
