"""Differential test: header and fence detection vs markdown-it-py on CommonMark examples.

Compares what the chunker decides with (top-level header lines and levels, fenced-code
line ranges) against markdown-it-py's CommonMark parser, on the spec's examples for ATX
and setext headings, fenced code and thematic breaks (data/, CC BY-SA 4.0).

Known divergences are listed with their reason and must keep diverging, so the list
stays accurate. Several are kept on purpose: changing them would change the chunks of
documents in the v4 study (its cached chunks are reproduced exactly).
"""
import json
from pathlib import Path

import pytest

markdown_it = pytest.importorskip("markdown_it")

from mdkeychunker.chunker import MarkdownChunker  # noqa: E402
from mdkeychunker.config import Config  # noqa: E402

DATA = json.loads((Path(__file__).parent / "data" / "commonmark-0.31.2-blocks.json")
                  .read_text(encoding="utf-8"))

KNOWN = {
    79: "empty ATX headings ('#', '## ') stay paragraphs (2 study documents contain them)",
    81: "multi-line setext headings are not detected (would change study documents)",
    82: "multi-line setext headings are not detected (would change study documents)",
    95: "multi-line setext headings are not detected (would change study documents)",
    85: "indented code blocks are not parsed (only fenced code is atomic)",
    100: "indented code blocks are not parsed (only fenced code is atomic)",
    93: "no lazy continuation into block quotes",
    96: "a leading '---' starts YAML front matter (extension)",
}


def _trim(lines: list[str], start: int, end: int) -> tuple[int, int]:
    while end > start and not lines[end].strip():  # an unclosed fence's trailing blank lines
        end -= 1
    return start, end


def reference(src: str) -> tuple[set, set]:
    lines = src.split("\n")
    heads, fences = set(), set()
    for t in markdown_it.MarkdownIt("commonmark").parse(src):
        if t.level != 0 or not t.map:
            continue
        if t.type == "heading_open":
            heads.add((t.map[0], int(t.tag[1])))
        elif t.type == "fence":
            fences.add(_trim(lines, t.map[0], t.map[1] - 1))
    return heads, fences


def ours(src: str) -> tuple[set, set]:
    lines = src.split("\n")
    heads, fences = set(), set()
    for b in MarkdownChunker(Config())._parse_blocks(src):
        if b.type == "header":
            heads.add((b.start_line, b.level))
        elif b.type == "code":
            fences.add(_trim(lines, b.start_line, b.end_line))
    return heads, fences


@pytest.mark.parametrize("ex", DATA["examples"], ids=lambda e: f"{e['example']}")
def test_header_and_fence_detection_matches_commonmark(ex):
    agree = reference(ex["markdown"]) == ours(ex["markdown"])
    if ex["example"] in KNOWN:
        assert not agree, f"#{ex['example']} now agrees; remove it from KNOWN"
    else:
        assert agree, f"#{ex['example']} ({ex['section']}): {ex['markdown']!r}"


def test_known_divergences_are_in_the_data():
    assert set(KNOWN) <= {e["example"] for e in DATA["examples"]}
