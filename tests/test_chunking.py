"""Tests for mk.chunk / mk.window / mk.chunk_file (0.4 free chunking)."""
import re
import warnings

import pytest

from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.chunking import HeaderlessWarning, chunk, chunk_file, window
from mdkeychunker.config import Config

DOC = """# Guide

Intro paragraph that explains what the guide covers and why it exists here.

## Install

Run the installer and follow the prompts. It takes a minute on most machines.

```bash
pip install example
```

### Linux

| distro | command |
|--------|---------|
| debian | apt install example |
| fedora | dnf install example |

## Usage

- first item in a list
- second item in a list

> A quoted note about usage that is long enough to stand on its own as a block.
"""


def words(s):
    return re.findall(r"\S+", s)


def lines_in_source(c, src):
    """Every non-blank line of the chunk text occurs in its source slice, in order."""
    window_src = src[c.start_char:c.end_char]
    pos = 0
    for line in (ln.strip() for ln in c.text.split("\n")):
        if not line:
            continue
        found = window_src.find(line, pos)
        if found < 0:
            return False
        pos = found + len(line)
    return True


# --------------------------------------------------------------------------- #
# Structure, prefix, section_path                                             #
# --------------------------------------------------------------------------- #

def test_uncapped_chunks_match_legacy_structural_chunker():
    # max_tokens=None must be the 0.3 / v4-study code path exactly
    legacy = MarkdownChunker(Config()).chunk(DOC)
    new = chunk(DOC, max_tokens=None)
    assert [c.text for c in new] == [c.text for c in legacy]
    assert [c.section_title for c in new] == [c.section_title for c in legacy]


def test_default_cap_leaves_small_documents_unchanged():
    assert [c.text for c in chunk(DOC)] == [c.text for c in MarkdownChunker(Config()).chunk(DOC)]


def test_embed_text_is_section_path_prefix():
    cs = chunk(DOC, doc_id="g")
    for c in cs:  # the study's render(mode="tc") layout: path, blank line, text
        assert c.embed_text == f"{' > '.join(c.section_path)}\n\n{c.text}"
    assert all(c.embed_text == c.text for c in chunk(DOC, prefix=False))


def test_section_path_of_merged_chunk_is_where_it_starts():
    cs = chunk("# A\r\n\r\nPara one.\r\n\r\n## B\r\n\r\nPara two with more words.\r\n")
    assert len(cs) == 1 and cs[0].section_path == ["A"]
    assert cs[0].embed_text.startswith("A\n\n# A")
    assert cs[0].section_title == "A > B"  # 0.3 value kept for compatibility
    assert cs[0].content_types == ["header", "paragraph"]


def test_section_path_skips_leading_headers_to_first_body_line():
    cs = chunk("# Guide\n\n## Install\n\nRun pip install.")
    assert cs[0].section_path == ["Guide", "Install"]


def test_no_prefix_without_section():
    cs = chunk("Just a paragraph with no header at all, nothing else.")
    assert cs[0].section_path == [] and cs[0].embed_text == cs[0].text


def test_section_path_keeps_titles_containing_separator():
    cs = chunk("# A > B\n\nbody text under a title with an arrow\n\n## C\n\nmore body text")
    assert cs[0].section_path == ["A > B"]  # one title, not ["A", "B"]
    assert cs[0].embed_text.startswith("A > B\n\n")


def test_section_path_nested():
    md = "# A\n\n" + "x " * 300 + "\n\n## B\n\n" + "y " * 300 + "\n\n# C\n\n" + "z " * 300
    assert [c.section_path for c in chunk(md)] == [["A"], ["A", "B"], ["C"]]


def test_empty_and_blank_input():
    assert chunk("") == [] and chunk(" \n\n\t") == []
    assert window("") == [] and window("   ") == []


def test_ids_links_positions_and_metadata():
    meta = {"source": "unit"}
    cs = chunk(DOC, doc_id="g", metadata=meta)
    assert [c.position_index for c in cs] == list(range(len(cs)))
    assert all(len(c.chunk_id) == 16 for c in cs)
    for a, b in zip(cs, cs[1:]):
        assert a.next_chunk_id == b.chunk_id and b.previous_chunk_id == a.chunk_id
    cs[0].metadata["x"] = 1
    assert meta == {"source": "unit"} and cs[1].metadata == {"source": "unit"}
    assert all(c.doc_id == "g" for c in cs)
    assert all(c.content_types == sorted(c.content_types) for c in cs)


def test_token_count_estimate_and_custom_tokenizer():
    cs = chunk(DOC)  # token_count is of embed_text, which the cap applies to
    assert all(c.token_count == -(-len(c.embed_text) // 4) for c in cs)
    cs = chunk(DOC, tokenizer=lambda s: len(words(s)))
    assert all(c.token_count == len(words(c.embed_text)) for c in cs)


def test_cap_applies_to_embed_text_including_prefix():
    md = "# " + " ".join(["Long section title"] * 6) + "\n\n" + "word " * 2000
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert all(len(c.embed_text) > len(c.text) for c in cs)
    assert all(-(-len(c.embed_text) // 4) <= 64 for c in cs)


def test_unknown_tokenizer_and_bad_cap():
    with pytest.raises(ValueError):
        chunk(DOC, tokenizer="sentencepiece")
    with pytest.raises(ValueError):
        chunk(DOC, max_tokens=0)


# --------------------------------------------------------------------------- #
# Offsets and stable ids                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_offsets_point_into_the_original_input(newline):
    src = DOC.replace("\n", newline)
    cs = chunk(src)
    assert all(lines_in_source(c, src) for c in cs)
    starts = [c.start_char for c in cs]
    assert starts == sorted(starts) and all(c.end_char > c.start_char for c in cs)
    assert src[cs[0].start_char:].startswith("# Guide")
    assert src[:cs[-1].end_char].rstrip().endswith("own as a block.")


def test_offsets_with_capped_splits():
    md = "# T\n\n" + " ".join(f"Sentence {i} is here." for i in range(400))
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert len(cs) > 5 and all(lines_in_source(c, md) for c in cs)
    for c in cs[1:]:
        assert md[c.start_char:c.end_char] == c.text  # prose pieces are exact slices


def test_ids_are_deterministic_and_stable_under_append():
    a = chunk(DOC, doc_id="g")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in chunk(DOC, doc_id="g")]
    b = chunk(DOC + "\n## Appendix\n\nA new section added at the very end of the guide.\n",
              doc_id="g")
    assert [c.chunk_id for c in b[:len(a) - 1]] == [c.chunk_id for c in a[:-1]]
    assert chunk(DOC, doc_id="h")[0].chunk_id != a[0].chunk_id


def test_ids_do_not_depend_on_enrichment_fields():
    c = chunk(DOC, doc_id="g")[0]
    before = c.chunk_id
    c.key, c.title, c.position_index = "k", "t", 99
    assert c.stable_id() == before


# --------------------------------------------------------------------------- #
# max_tokens and oversized splitting                                          #
# --------------------------------------------------------------------------- #

def test_oversized_code_block_is_split_line_aware_and_refenced():
    body = "\n".join(f"line_{i} = compute({i})  # comment" for i in range(600))
    md = f"# Code\n\n```python\n{body}\n```\n"
    cs = chunk(md, max_tokens=128, headerless="structure")
    assert len(cs) > 10
    assert all(c.oversized and c.token_count <= 128 for c in cs)
    assert all(c.section_path == ["Code"] for c in cs)
    rebuilt = []
    for i, c in enumerate(cs):
        code = c.text.split("```python\n", 1)
        assert len(code) == 2 and c.text.endswith("\n```")
        assert (code[0] == "# Code\n\n") == (i == 0)  # header only on the first piece
        rebuilt.extend(code[1][:-4].split("\n"))
    assert rebuilt == body.split("\n")  # every line exactly once, in order, never cut


def test_unclosed_fence_pieces_get_closed():
    body = "\n".join(f"x{i} = {i}" for i in range(400))
    cs = chunk(f"~~~~\n{body}\n", max_tokens=64, headerless="structure")
    assert len(cs) > 1
    assert all(c.text.startswith("~~~~\n") and c.text.endswith("\n~~~~") for c in cs)


def test_oversized_table_repeats_header_row():
    rows = [f"| r{i} | value number {i} |" for i in range(300)]
    md = "## T\n\n| key | value |\n|-----|-------|\n" + "\n".join(rows)
    cs = chunk(md, max_tokens=100, headerless="structure")
    assert len(cs) > 3 and all(c.oversized and c.token_count <= 100 for c in cs)
    got = []
    for c in cs:
        table = c.text.split("| key | value |\n|-----|-------|\n")
        assert len(table) == 2
        got.extend(table[1].split("\n"))
    assert got == rows


def test_oversized_paragraph_splits_at_sentences():
    sents = [f"This is sentence number {i} of a very long paragraph." for i in range(200)]
    md = "# P\n\n" + " ".join(sents)
    cs = chunk(md, max_tokens=80, headerless="structure")
    assert all(c.oversized and c.token_count <= 80 for c in cs)
    assert words(" ".join(c.text for c in cs)) == words(md)
    assert all(c.text.endswith(".") for c in cs)  # no sentence cut while sentences fit


def test_unbroken_text_is_hard_cut_losslessly():
    blob = "A" * 10_000
    cs = chunk(f"# B\n\n{blob}", max_tokens=50, headerless="structure")
    assert all(c.token_count <= 50 for c in cs)
    assert "".join(c.text for c in cs).replace("# B\n\n", "") == blob


def test_overcap_chunk_of_small_blocks_is_repacked_without_flag():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(30)]
    md = "# S\n\n" + "\n\n".join(paras)
    # max_chars keeps one structural chunk, which is over the token cap
    cs = chunk(md, max_tokens=200, max_chars=100_000, headerless="structure")
    assert len(cs) > 1 and not any(c.oversized for c in cs)
    assert all(c.token_count <= 200 for c in cs)
    assert words(" ".join(c.text for c in cs)) == words(md)


def test_no_chunk_ends_with_a_header_after_capping():
    md = "\n\n".join(f"## H{i}\n\n" + "body " * 120 for i in range(10))
    for c in chunk(md, max_tokens=64, max_chars=100_000, headerless="structure"):
        last = [ln for ln in c.text.split("\n") if ln.strip()][-1]
        assert not last.startswith("#")


def test_header_counts_toward_first_piece_budget():
    md = "# A fairly long section header for budget\n\n" + "word " * 2000
    assert all(c.token_count <= 100 for c in chunk(md, max_tokens=100, headerless="structure"))


# --------------------------------------------------------------------------- #
# Headerless fallback                                                         #
# --------------------------------------------------------------------------- #

PLAIN = "\n\n".join(f"Plain paragraph {i}. " + "text " * 60 for i in range(40))


def test_headerless_input_falls_back_to_windows_with_warning():
    with pytest.warns(HeaderlessWarning):
        cs = chunk(PLAIN, doc_id="p")
    assert all(c.content_types == ["window"] and c.token_count <= 256 for c in cs)
    assert all(PLAIN[c.start_char:c.end_char] == c.text for c in cs)
    assert all(c.doc_id == "p" for c in cs)


def test_headerless_structure_keeps_paragraph_grouping():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cs = chunk(PLAIN, headerless="structure")
    assert all("window" not in c.content_types for c in cs)


def test_short_or_structured_docs_do_not_fall_back():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunk("A short note without headers.")
        chunk(DOC)


def test_one_header_with_a_table_is_structure_not_windows():
    # lead's repro: a table doc with one header must not be windowed through its rows
    rows = [f"| {i} | value {i} with words |" for i in range(400)]
    md = "# T\n\n| a | b |\n|---|---|\n" + "\n".join(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cs = chunk(md, doc_id="t")
    assert len(cs) > 1 and all(c.text.startswith("| a | b |\n|---|---|\n") for c in cs[1:])
    got = [ln for c in cs for ln in c.text.split("\n") if re.match(r"^\| \d+ \|", ln)]
    assert got == rows and all(c.token_count <= 512 for c in cs)


def test_two_headers_far_apart_is_structure():
    md = "# One\n\n" + PLAIN + "\n\n# Two\n\n" + PLAIN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cs = chunk(md)
    assert all("window" not in c.content_types for c in cs)
    assert {tuple(c.section_path) for c in cs} == {("One",), ("Two",)}


def test_one_header_plain_prose_is_windowed_with_its_path():
    md = "# Notes\n\n" + PLAIN
    with pytest.warns(HeaderlessWarning):
        cs = chunk(md)
    assert all(c.content_types == ["window"] for c in cs)
    assert cs[-1].section_path == ["Notes"] and cs[-1].embed_text.startswith("Notes\n\n")
    assert all(c.token_count <= 512 for c in cs)


def test_headerless_windows_never_cut_tables_or_fences():
    code = "\n".join(f"line_{i} = {i}" for i in range(300))
    table = "| k | v |\n|---|---|\n" + "\n".join(f"| r{i} | v{i} |" for i in range(300))
    md = PLAIN + "\n\n```py\n" + code + "\n```\n\n" + PLAIN + "\n\n" + table + "\n\n" + PLAIN
    with pytest.warns(HeaderlessWarning):
        cs = chunk(md, max_tokens=256)
    kinds = {t for c in cs for t in c.content_types}
    assert kinds == {"window", "code", "table"}
    for c in cs:
        if "code" in c.content_types:
            assert c.text.startswith("```py\n") and c.text.endswith("\n```")
        if "table" in c.content_types:
            assert c.text.startswith("| k | v |\n|---|---|\n")
        if c.content_types == ["window"]:
            assert "```" not in c.text and "|---|" not in c.text
    assert all(c.token_count <= 256 for c in cs)
    assert_tiles(cs, md, overlap=True)


def test_headerless_window_forced():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cs = chunk(DOC, headerless="window")
    assert {t for c in cs for t in c.content_types} == {"window", "code", "table"}


# --------------------------------------------------------------------------- #
# window                                                                      #
# --------------------------------------------------------------------------- #

def test_window_exact_slices_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    cs = window(text, size=50, overlap=10, tokenizer=lambda s: len(words(s)))
    assert all(text[c.start_char:c.end_char] == c.text for c in cs)
    assert all(len(words(c.text)) <= 50 for c in cs)
    for a, b in zip(cs, cs[1:]):
        assert words(a.text)[-10:] == words(b.text)[:10]
    assert words(cs[-1].text)[-1] == "w999"


def test_window_without_overlap_is_a_partition():
    text = "alpha beta gamma delta " * 200
    cs = window(text, size=32, overlap=0)
    assert words(" ".join(c.text for c in cs)) == words(text)
    assert all(c.token_count <= 32 for c in cs)


def test_window_long_word_and_validation():
    cs = window("x" * 5000, size=100, overlap=0)
    assert "".join(c.text for c in cs) == "x" * 5000
    for size, overlap in [(0, 0), (10, 10), (10, -1)]:
        with pytest.raises(ValueError):
            window("a b c", size=size, overlap=overlap)
    with pytest.raises(ValueError):
        chunk("x", headerless="windows")


def test_window_with_tiktoken_matches_token_budget():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")
    text = DOC * 20 + " 日本語のテキスト" * 50
    for tok in ("tiktoken", enc):
        cs = window(text, size=64, overlap=8, tokenizer=tok)
        assert all(text[c.start_char:c.end_char] == c.text for c in cs)
        assert all(len(enc.encode(c.text)) <= 64 for c in cs)
        assert cs[0].token_count == len(enc.encode(cs[0].text))


# --------------------------------------------------------------------------- #
# chunk_file                                                                  #
# --------------------------------------------------------------------------- #

def test_chunk_file_reads_markdown_and_sets_doc_id(tmp_path):
    p = tmp_path / "guide.md"
    p.write_text(DOC, encoding="utf-8")
    cs = chunk_file(p)
    assert cs[0].doc_id == p.as_posix() and [c.text for c in cs] == [c.text for c in chunk(DOC)]
    assert chunk_file(str(p), doc_id="x")[0].doc_id == "x"
    assert chunk_file(p, max_tokens=16)[0].token_count <= 16


def test_chunk_file_offsets_match_crlf_file(tmp_path):
    p = tmp_path / "crlf.md"
    p.write_bytes(DOC.replace("\n", "\r\n").encode("utf-8"))
    raw = p.read_bytes().decode("utf-8")
    assert all(lines_in_source(c, raw) for c in chunk_file(p))


def test_chunk_file_rejects_binary_formats(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.7")
    with pytest.raises(ValueError, match="Docling"):
        chunk_file(p)


# --------------------------------------------------------------------------- #
# Regressions found by fuzzing random Markdown                                #
# --------------------------------------------------------------------------- #

def assert_tiles(cs, md, overlap=False):
    """Chunk spans cover the document in order; gaps hold only whitespace."""
    prev = start = 0
    for c in cs:
        assert c.start_char >= (start if overlap else prev)
        assert not md[prev:c.start_char].strip()
        start, prev = c.start_char, max(prev, c.end_char)
    assert not md[prev:].strip()


def test_header_line_over_the_cap_is_split():
    md = "# " + "longword " * 400 + "\n\nbody text"
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert all(c.token_count <= 64 for c in cs) and any(c.oversized for c in cs)
    assert_tiles(cs, md)


def test_header_run_too_long_to_travel_keeps_the_cap():
    md = "\n\n".join(f"### heading {i} " + "w" * 40 for i in range(8)) + "\n\n" + "body " * 400
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert all(c.token_count <= 64 for c in cs)
    assert_tiles(cs, md)


def test_trailing_headers_do_not_overflow_the_last_chunk():
    md = "# A\n\n" + "body " * 400 + "\n\n## Empty one\n\n## Empty two"
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert all(c.token_count <= 64 for c in cs)
    assert cs[-1].text.endswith("## Empty two")
    assert_tiles(cs, md)


def test_table_whose_first_row_is_over_the_cap_keeps_header_and_offsets():
    md = "| k | v |\n|---|---|\n| " + "cell " * 300 + "| x |\n| a | b |"
    cs = chunk(md, max_tokens=64, headerless="structure")
    assert all(c.text.startswith("| k | v |\n|---|---|\n") for c in cs)
    assert all(c.token_count <= 64 for c in cs) and cs[0].start_char == 0
    assert_tiles(cs, md)


def test_capped_chunks_tile_the_document():
    for md in (DOC * 5, "# T\n\n" + " ".join(f"Sentence {i}." for i in range(500))):
        assert_tiles(chunk(md, max_tokens=48, headerless="structure"), md)
