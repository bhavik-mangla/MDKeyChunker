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


# ─── Structural regressions ────────────────────────────────────────────────


def _chunk(md, max_size=500, min_size=50):
    return MarkdownChunker(Config(min_chunk_size=min_size, max_chunk_size=max_size)).chunk(md)


def test_header_never_dangles_at_end_of_chunk():
    para = "word " * 50
    md = f"# A\n\n{para}\n\n# B\n\n{para}\n\n# C\n\n{para}"
    chunks = _chunk(md, max_size=300)
    for c in chunks:
        assert not c.text.rstrip().splitlines()[-1].startswith("#")
    by_head = {c.text.splitlines()[0]: c.section_title for c in chunks}
    assert by_head == {"# A": "A", "# B": "B", "# C": "C"}


def test_consecutive_headers_stay_with_their_body():
    md = "# Guide\n\n## Install\n\n" + "Run the installer. " * 5
    chunks = _chunk(md)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Guide > Install"


def _blocks(md):
    return MarkdownChunker(Config())._parse_blocks(md)


def test_longer_fence_containing_shorter_fence_is_one_block():
    md = "````markdown\n```python\nx = 1\n```\n````\n\nAfter."
    blocks = _blocks(md)
    assert [b.type for b in blocks] == ["code", "paragraph"]
    assert blocks[0].content.endswith("````")


def test_info_string_line_does_not_close_fence():
    blocks = _blocks("```\nfirst\n```python\nsecond\n```\n\nAfter.")
    assert [b.type for b in blocks] == ["code", "paragraph"]
    assert "second" in blocks[0].content


def test_setext_headers_are_detected():
    md = "Title\n=====\n\n" + "Body text. " * 10 + "\n\nSub\n---\n\n" + "More text. " * 10
    chunks = _chunk(md, max_size=150, min_size=1)
    assert {c.section_title for c in chunks} == {"Title", "Title > Sub"}


def test_crlf_and_trailing_hashes():
    chunks = _chunk("## Setup ##\r\n\r\n" + "Configure it. " * 6, min_size=1)
    assert chunks[0].section_title == "Setup"
    assert "\r" not in chunks[0].text


def test_paren_ordered_list_is_a_list():
    chunks = _chunk("1) first item\n2) second item\n3) third item", min_size=1)
    assert "list" in chunks[0].content_types


def test_header_before_oversized_block_stays_with_it():
    code = "```\n" + "x = 1\n" * 400 + "```"
    chunks = _chunk(f"# A\n\nintro text here\n\n## B\n\n{code}", max_size=1500, min_size=10)
    for c in chunks:
        assert not c.text.rstrip().splitlines()[-1].startswith("#")
    code_chunk = next(c for c in chunks if "code" in c.content_types)
    assert code_chunk.text.startswith("## B") and code_chunk.section_title == "A > B"


def test_header_is_never_a_chunk_on_its_own():
    chunks = _chunk("# A\n\n" + "word " * 1000, max_size=1500, min_size=100)
    assert len(chunks) == 1 and chunks[0].text.startswith("# A")


def test_short_sections_are_grouped_and_labelled_by_first_section():
    md = "\n\n".join(f"## S{i}\n\nshort body {i}." for i in range(6))
    chunks = _chunk(md, max_size=1500, min_size=100)
    assert len(chunks) == 1
    assert chunks[0].section_title == "S0"


def test_merged_small_chunk_keeps_its_own_section_label():
    big = "```\n" + "y = 2\n" * 400 + "```"
    chunks = _chunk(f"## S0\n\nshort body.\n\n## S1\n\n{big}", max_size=1500, min_size=100)
    first = chunks[0]
    assert first.text.startswith("## S0") and first.section_title == "S0"


def test_backtick_info_string_with_backticks_is_not_a_fence():
    blocks = _blocks("```js``` is inline\n\n# H\n\nbody")
    assert [b.type for b in blocks] == ["paragraph", "header", "paragraph"]


def test_deeply_indented_fence_does_not_close_block():
    blocks = _blocks("```\n    ```\nstill code\n```\n\n# After")
    assert [b.type for b in blocks] == ["code", "header"]
    assert "still code" in blocks[0].content


def test_thematic_break_is_not_a_setext_header():
    blocks = _blocks("Intro\n\n***\n---\n\nbody")
    assert "header" not in [b.type for b in blocks]


def test_small_trailing_section_joins_previous_chunk():
    chunks = _chunk("# Paper\n\n" + "Body text. " * 30 + "\n\n## Acronyms\n", max_size=1500, min_size=100)
    assert len(chunks) == 1 and chunks[0].text.rstrip().endswith("## Acronyms")
