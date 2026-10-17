"""Property tests (Hypothesis) for mk.chunk invariants (RESEARCH.md sec. 4).

Markdown is generated from a small grammar: ATX and setext headers, fences of varying
character and length (the last one possibly unclosed), tables, lists, blockquotes,
paragraphs with very long words, LF or CRLF. Expectations come from an independent
line tracker below, not from the chunker's own parser.
"""
import math
import os
import re

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import mdkeychunker as mk  # noqa: E402

# MDK_HYPOTHESIS_EXAMPLES=1000 for a longer run
SETTINGS = settings(max_examples=int(os.environ.get("MDK_HYPOTHESIS_EXAMPLES", "150")),
                    deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

ATX = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)\s*$")

# --------------------------------------------------------------------------- #
# Grammar                                                                     #
# --------------------------------------------------------------------------- #

WORDS = st.sampled_from(["alpha", "beta", "gamma.", "delta!", "x=1", "a>b", "end?", "日本語"])
word = st.one_of(WORDS, st.integers(1, 120).map(lambda n: "y" * n))
text = st.lists(word, min_size=1, max_size=40).map(" ".join)
title = st.lists(WORDS, min_size=1, max_size=4).map(" ".join).filter(
    lambda t: not t.startswith(("-", "*", "+", ">", "#")) and not re.match(r"^\d+[.)]", t))


@st.composite
def atx(draw):
    return "#" * draw(st.integers(1, 6)) + " " + draw(title)


@st.composite
def setext(draw):
    return draw(title) + "\n" + draw(st.sampled_from("=-")) * draw(st.integers(1, 6))


@st.composite
def paragraph(draw):
    return "\n".join(draw(st.lists(text, min_size=1, max_size=4)))


@st.composite
def fence(draw, closed=True):
    char = draw(st.sampled_from("`~"))
    n = draw(st.integers(3, 5))
    body = draw(st.lists(st.one_of(text, st.just("# not a header"), st.just(char * (n - 1)),
                                   st.just("")), max_size=60))
    block = [char * n + draw(st.sampled_from(["", "py", "bash"]))] + body
    return "\n".join(block + ([char * n] if closed else []))


@st.composite
def table(draw):
    rows = draw(st.lists(st.tuples(text, text), min_size=0, max_size=60))
    return "\n".join(["| key | value |", "|-----|-------|"] + [f"| {a} | {b} |" for a, b in rows])


@st.composite
def bullet_list(draw):
    return "\n".join(f"- {t}" for t in draw(st.lists(text, min_size=1, max_size=15)))


@st.composite
def quote(draw):
    return "\n".join(f"> {t}" for t in draw(st.lists(text, min_size=1, max_size=5)))


block = st.one_of(atx(), atx(), setext(), paragraph(), paragraph(), fence(), table(),
                  bullet_list(), quote())


@st.composite
def markdown(draw):
    blocks = draw(st.lists(block, min_size=1, max_size=25))
    if draw(st.booleans()):
        blocks.append(draw(fence(closed=False)))  # an unclosed fence runs to the end
    md = "\n\n".join(blocks) + "\n"
    return md.replace("\n", "\r\n") if draw(st.booleans()) else md


CAPS = st.sampled_from([48, 96, 256, 512])

# --------------------------------------------------------------------------- #
# Reference tracker                                                           #
# --------------------------------------------------------------------------- #


def scan(md: str):
    """Per line: (start offset, line, is_header_line, header stack, inside a fence)."""
    out, stack, fence_open = [], [], None
    lines = md.replace("\r\n", "\n").split("\n")
    starts, pos = [], 0
    for raw in re.split(r"(?<=\n)", md):
        starts.append(pos)
        pos += len(raw)
    for i, line in enumerate(lines):
        in_fence = fence_open is not None
        is_header = False
        if fence_open:
            if re.match(rf"^ {{0,3}}{re.escape(fence_open[0])}{{{fence_open[1]},}}\s*$", line):
                fence_open = None
        elif m := re.match(r"^ {0,3}(`{3,}|~{3,})", line):
            fence_open, in_fence = (m.group(1)[0], len(m.group(1))), True
        elif m := ATX.match(line):
            level = len(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2).strip())]
            is_header = True
        elif (line.strip() and i + 1 < len(lines) and SETEXT_UNDERLINE.match(lines[i + 1])
              and not line.startswith(("|", "-", ">"))):
            level = 1 if lines[i + 1].strip()[0] == "=" else 2
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, line.strip())]
            is_header = True
        elif out and out[-1][2] and SETEXT_UNDERLINE.match(line) and i > 0 \
                and not ATX.match(lines[i - 1]):
            is_header = True  # the underline of the setext header above
        out.append((starts[i] if i < len(starts) else len(md), line, is_header,
                    [t for _, t in stack], in_fence))
    return out


def nonspace(s: str) -> str:
    return re.sub(r"\s", "", s)


def is_subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def _ends_closed(chunk_text: str) -> bool:
    fence_open = None
    for line in chunk_text.split("\n"):
        if fence_open:
            if re.match(rf"^ {{0,3}}{re.escape(fence_open[0])}{{{fence_open[1]},}}\s*$", line):
                fence_open = None
        elif m := re.match(r"^ {0,3}(`{3,}|~{3,})", line):
            fence_open = (m.group(1)[0], len(m.group(1)))
    return fence_open is None


# --------------------------------------------------------------------------- #
# Invariants                                                                  #
# --------------------------------------------------------------------------- #

@SETTINGS
@given(markdown(), CAPS)
def test_lossless_ordered_and_tiled(md, cap):
    """Inv. 1: every character of the input is in some chunk, in order; spans tile the doc."""
    cs = mk.chunk(md, max_tokens=cap, headerless="structure")
    assert is_subsequence(nonspace(md), nonspace("".join(c.text for c in cs)))
    prev = 0
    for c in cs:
        assert c.start_char >= prev and not md[prev:c.start_char].strip()
        prev = c.end_char
    assert not md[prev:].strip()


@SETTINGS
@given(markdown(), CAPS)
def test_fences_are_never_left_open(md, cap):
    """Inv. 2: no chunk splits a fence; split code pieces are re-fenced.

    The one exception is a fence the input itself never closes, in the chunk that ends the
    document.
    """
    cs = mk.chunk(md, max_tokens=cap, headerless="structure")
    doc_closed = _ends_closed(md.replace("\r\n", "\n"))
    for c in cs[:-1] if not doc_closed else cs:
        assert _ends_closed(c.text), c.text[-200:]


@SETTINGS
@given(markdown(), CAPS)
def test_no_chunk_ends_with_a_header(md, cap):
    """Inv. 3. Exceptions, both documented: a chunk holding only headers (a header run too
    long to travel with its content), and empty sections at the very end of the document,
    which join the last chunk (0.3 behaviour)."""
    for c in mk.chunk(md, max_tokens=cap, headerless="structure"):
        rows = [r for r in scan(c.text) if r[1].strip()]
        if not rows or all(r[2] for r in rows) or not md[c.end_char:].strip():
            continue
        assert not rows[-1][2], c.text[-200:]


@pytest.mark.filterwarnings("ignore::mdkeychunker.chunking.HeaderlessWarning")
@SETTINGS
@given(markdown(), CAPS, st.sampled_from(["structure", "auto"]))
def test_embed_text_never_exceeds_the_cap(md, cap, mode):
    """Inv. 4, stricter than planned: embed_text (prefix included) <= max_tokens, always."""
    for c in mk.chunk(md, max_tokens=cap, headerless=mode):
        assert c.token_count == math.ceil(len(c.embed_text) / 4) <= cap
        assert c.oversized or c.text in (c.embed_text, c.embed_text.split("\n\n", 1)[-1])


@SETTINGS
@given(markdown())
def test_section_path_is_the_header_stack_at_the_first_body_line(md):
    """Inv. 5 (on uncapped chunks; capped pieces inherit their block's path)."""
    rows = scan(md)
    for c in mk.chunk(md, max_tokens=None, headerless="structure"):
        mine = [r for r in rows if c.start_char <= r[0] < c.end_char and r[1].strip()]
        body = [r for r in mine if not r[2]]
        expected = body[0][3] if body else mine[-1][3]
        assert c.section_path == expected


@SETTINGS
@given(markdown(), CAPS)
def test_ids_are_deterministic_and_stable_under_append(md, cap):
    """Inv. 6. The last chunk may change: a small one merges with the appended section."""
    a = mk.chunk(md, doc_id="d", max_tokens=cap, headerless="structure")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in
                                       mk.chunk(md, doc_id="d", max_tokens=cap,
                                                headerless="structure")]
    if not _ends_closed(md.replace("\r\n", "\n")):
        return  # an unclosed fence swallows anything appended
    b = mk.chunk(md + "\n# Appendix\n\nA section appended at the end of the document.\n",
                 doc_id="d", max_tokens=cap, headerless="structure")
    assert [c.chunk_id for c in b[:len(a) - 1]] == [c.chunk_id for c in a[:-1]]


@SETTINGS
@given(markdown())
def test_rechunking_a_single_section_chunk_is_idempotent(md):
    """Inv. 7, scoped: a chunk with at most one header and no split comes back whole."""
    for c in mk.chunk(md, headerless="structure"):
        headers = sum(r[2] for r in scan(c.text) if r[1].strip())
        if c.oversized or headers > 1:
            continue
        again = mk.chunk(c.text, headerless="structure")
        assert [x.text for x in again] == [c.text]
