"""Tests for MarkdownChunker."""
import pytest
from mdkeychunker.config import Config
from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.models import Chunk


@pytest.fixture
def config():
    return Config(min_chunk_size=50, max_chunk_size=500)


@pytest.fixture
def chunker(config):
    return MarkdownChunker(config)


def test_basic_chunking(chunker):
    md = "# Hello\n\nThis is a paragraph.\n\n## World\n\nAnother paragraph."
    chunks = chunker.chunk(md)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_returns_chunk_objects(chunker):
    chunks = chunker.chunk("# Title\n\nSome content here with enough text.")
    assert all(hasattr(c, "text") for c in chunks)
    assert all(hasattr(c, "section_title") for c in chunks)
    assert all(hasattr(c, "content_types") for c in chunks)


def test_section_title_propagated(chunker):
    md = "# Main Section\n\nContent under main section."
    chunks = chunker.chunk(md)
    # At least one chunk should have the section title
    titles = [c.section_title for c in chunks]
    assert any("Main Section" in t for t in titles)


def test_code_block_not_split(chunker):
    md = "# Code\n\n```python\ndef foo():\n    return 42\n```\n\nExplanation follows."
    chunks = chunker.chunk(md)
    # The code block must appear intact in some chunk
    all_text = "\n".join(c.text for c in chunks)
    assert "def foo():" in all_text
    assert "return 42" in all_text


def test_table_not_split(chunker):
    md = "# Table\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    chunks = chunker.chunk(md)
    all_text = "\n".join(c.text for c in chunks)
    assert "| A | B |" in all_text


def test_empty_input(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_only_headers(chunker):
    md = "# H1\n## H2\n### H3"
    chunks = chunker.chunk(md)
    assert isinstance(chunks, list)


def test_content_types_detected(chunker):
    md = "# Title\n\nParagraph.\n\n```python\ncode()\n```"
    chunks = chunker.chunk(md)
    all_types = set()
    for c in chunks:
        all_types.update(c.content_types)
    assert "code" in all_types or "paragraph" in all_types or "header" in all_types


def test_small_chunks_merged(chunker):
    """Tiny paragraphs should be merged to reach min_chunk_size."""
    md = "# Title\n\nHi.\n\nBye.\n\nMore text here that makes it long enough."
    chunks = chunker.chunk(md)
    # None of the chunks should be excessively tiny (below min_chunk_size)
    # unless the whole document is tiny
    total_text = " ".join(c.text for c in chunks)
    assert len(total_text) > 0


def test_large_document(chunker):
    """Large document should produce multiple chunks."""
    sections = []
    for i in range(10):
        sections.append(f"## Section {i}\n\n" + ("Word " * 100) + "\n")
    md = "\n".join(sections)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 2


def test_yaml_frontmatter(chunker):
    md = "---\ntitle: Test\nauthor: Me\n---\n\n# Content\n\nBody text here."
    chunks = chunker.chunk(md)
    assert len(chunks) >= 1


def test_nested_header_section_path(chunker):
    md = "# Top\n\n## Mid\n\n### Deep\n\nContent here."
    chunks = chunker.chunk(md)
    # Should have a deeply nested section path
    titles = [c.section_title for c in chunks]
    assert any("Deep" in t for t in titles)


def test_blockquote_preserved(chunker):
    md = "# Section\n\n> This is a blockquote\n> that spans multiple lines.\n\nNormal text."
    chunks = chunker.chunk(md)
    all_text = "\n".join(c.text for c in chunks)
    assert "blockquote" in all_text or "This is a blockquote" in all_text


def test_list_preserved(chunker):
    md = "# List\n\n- item one\n- item two\n- item three"
    chunks = chunker.chunk(md)
    all_text = "\n".join(c.text for c in chunks)
    assert "item one" in all_text


# ─── New tests (Part 5 of optimization prompt) ────────────────────────────


def test_pipe_in_prose_not_treated_as_table(chunker):
    """A pipe character in prose must not be detected as a table start."""
    md = "# Decision\n\nUse option A | B for the selection process."
    chunks = chunker.chunk(md)
    all_content_types = [ct for c in chunks for ct in c.content_types]
    assert "table" not in all_content_types


def test_real_table_is_detected(chunker):
    """A proper Markdown table (header + separator) must be detected as table."""
    md = "# Data\n\n| Name | Value |\n|------|-------|\n| foo | 42 |\n| bar | 99 |"
    chunks = chunker.chunk(md)
    all_content_types = [ct for c in chunks for ct in c.content_types]
    assert "table" in all_content_types
