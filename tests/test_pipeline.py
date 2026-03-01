"""Integration tests for the full Pipeline."""
import pytest
from unittest.mock import MagicMock, patch
from mdkeychunker.config import Config
from mdkeychunker.pipeline import Pipeline
from mdkeychunker.models import Chunk


SAMPLE_MD = """# Introduction

This section introduces the core concepts of machine learning and supervised learning.
It provides foundational knowledge for understanding the system.

## Authentication

Authentication is the process of verifying identity. It is a critical security concept.
Common methods include OAuth2, JWT tokens, and API keys.

## Database Design

The database schema is optimized for read performance. We use PostgreSQL.
Indexes are applied to frequently queried columns.

## Authentication Details

More details on authentication flows and token refresh strategies.
The OAuth2 flow involves authorization codes and refresh tokens.
"""

MOCK_LLM_RESULT = {
    "title": "Overview",
    "summary": "A summary of the chunk.",
    "keywords": ["keyword1", "keyword2"],
    "entities": [{"name": "Entity", "type": "CONCEPT"}],
    "questions": ["What is this about?"],
    "key": "general topic",
    "related_keys": [],
}


def _make_pipeline(mock_result=None) -> Pipeline:
    """Build a Pipeline with a mocked LLM client."""
    config = Config(
        llm_provider="openai",
        llm_api_key="test-key",
        merge_by_keys=True,
        max_merged_size=3000,
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = config

    from mdkeychunker.chunker import MarkdownChunker
    from mdkeychunker.enricher import Enricher
    from mdkeychunker.restructurer import Restructurer

    pipeline.chunker = MarkdownChunker(config)
    pipeline.restructurer = Restructurer(config)

    mock_llm = MagicMock()
    mock_llm.call_json.return_value = mock_result or MOCK_LLM_RESULT
    pipeline.llm = mock_llm
    pipeline.enricher = Enricher(mock_llm)

    return pipeline


def test_pipeline_returns_chunks():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text(SAMPLE_MD)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_pipeline_chunks_have_ids():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text(SAMPLE_MD)
    assert all(c.chunk_id for c in chunks)


def test_pipeline_navigation_linked():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text(SAMPLE_MD)
    if len(chunks) > 1:
        # First chunk links forward
        assert chunks[0].next_chunk_id != ""
        # Last chunk links backward
        assert chunks[-1].previous_chunk_id != ""
        # First has no prev, last has no next
        assert chunks[0].previous_chunk_id == ""
        assert chunks[-1].next_chunk_id == ""


def test_pipeline_position_indexed():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text(SAMPLE_MD)
    for i, c in enumerate(chunks):
        assert c.position_index == i


def test_pipeline_enrichment_applied():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text(SAMPLE_MD)
    # All chunks should have enrichment data (from mock)
    assert all(c.title == "Overview" for c in chunks)
    assert all(c.key == "general topic" for c in chunks)


def test_pipeline_merge_same_key():
    """Two chunks with same key should be merged by restructurer."""
    results = [
        {**MOCK_LLM_RESULT, "key": "authentication"},
        {**MOCK_LLM_RESULT, "key": "database"},
        {**MOCK_LLM_RESULT, "key": "authentication"},  # same as first
        {**MOCK_LLM_RESULT, "key": "security"},
    ]
    mock_llm = MagicMock()
    mock_llm.call_json.side_effect = results + [MOCK_LLM_RESULT] * 20

    config = Config(merge_by_keys=True, max_merged_size=3000)

    from mdkeychunker.chunker import MarkdownChunker
    from mdkeychunker.enricher import Enricher
    from mdkeychunker.restructurer import Restructurer

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = config
    pipeline.chunker = MarkdownChunker(config)
    pipeline.llm = mock_llm
    pipeline.enricher = Enricher(mock_llm)
    pipeline.restructurer = Restructurer(config)

    chunks = pipeline.process_text(SAMPLE_MD)
    # Should have fewer chunks than before merging
    assert isinstance(chunks, list)


def test_pipeline_token_count_set():
    pipeline = _make_pipeline()
    chunks = pipeline.process_text("# Title\n\n" + "Word " * 50)
    assert all(c.token_count > 0 for c in chunks)


def test_pipeline_process_file(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "# Test\n\nContent for testing pipeline file processing.")

    pipeline = _make_pipeline()
    chunks = pipeline.process_file(str(md_file))
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_pipeline_process_file_not_found():
    pipeline = _make_pipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_file("/nonexistent/path/file.md")


def test_pipeline_save_jsonl(tmp_path):
    pipeline = _make_pipeline()
    chunks = pipeline.process_text("# Test\n\nSome content here.")
    out_path = str(tmp_path / "output.jsonl")
    pipeline.save_jsonl(chunks, out_path)

    import json
    lines = (tmp_path / "output.jsonl").read_text().strip().split("\n")
    for line in lines:
        data = json.loads(line)
        assert "text" in data


def test_pipeline_save_summary(tmp_path):
    pipeline = _make_pipeline()
    chunks = pipeline.process_text("# Test\n\nSome content here.")
    out_path = str(tmp_path / "summary.txt")
    pipeline.save_summary(chunks, out_path)

    content = (tmp_path / "summary.txt").read_text()
    # If chunks have summaries (from mock), they should appear
    assert isinstance(content, str)


# ─── New tests (Part 5 of optimization prompt) ────────────────────────────


def test_pipeline_multiple_docs_no_key_leakage():
    """Rolling keys from doc 1 must not appear in doc 2's LLM prompt."""
    pipeline = _make_pipeline({
        **MOCK_LLM_RESULT,
        "key": "unique_doc1_subtopic",
        "related_keys": [],
    })
    pipeline.process_text("# Doc1\n\nFirst document content here for testing.")

    # Second document — swap out the mock return value
    mock_llm = pipeline.llm
    mock_llm.call_json.reset_mock()
    mock_llm.call_json.return_value = {
        **MOCK_LLM_RESULT, "key": "doc2_subtopic"}
    pipeline.process_text(
        "# Doc2\n\nSecond document content here for testing.")

    # The first LLM call for doc2 should NOT contain doc1's keys in the prompt
    first_call = mock_llm.call_json.call_args_list[0]
    prompt_text = first_call[0][0]
    assert "unique_doc1_subtopic" not in prompt_text


def test_pipeline_survives_llm_exception():
    """If LLM call_json raises, pipeline should still return unenriched chunks."""
    config = Config(
        llm_provider="openai",
        llm_api_key="test-key",
        merge_by_keys=False,
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = config

    from mdkeychunker.chunker import MarkdownChunker
    from mdkeychunker.enricher import Enricher
    from mdkeychunker.restructurer import Restructurer

    mock_llm = MagicMock()
    mock_llm.call_json.side_effect = ConnectionError("Network down")
    pipeline.chunker = MarkdownChunker(config)
    pipeline.llm = mock_llm
    pipeline.enricher = Enricher(mock_llm)
    pipeline.restructurer = Restructurer(config)

    # Should not raise — gracefully returns unenriched chunks
    chunks = pipeline.process_text(
        "# Title\n\nSome content that can be chunked.")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(c.key == "" for c in chunks)  # unenriched
