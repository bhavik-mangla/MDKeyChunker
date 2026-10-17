"""Free, structure-first chunking: ``chunk``, ``chunk_file`` and ``window``.

``chunk`` cuts Markdown at headers (the 0.3 structural chunker, unchanged) and then caps
every chunk's ``embed_text`` at ``max_tokens``: a chunk over the cap is re-packed at block
boundaries, and an atomic block that alone exceeds it (code, table, a huge paragraph) is
split inside — line-aware and re-fenced for code, row-aware with the header row repeated
for tables, paragraph/line/sentence/word for prose — and flagged ``oversized=True``.

``Chunk.text`` stays raw; ``Chunk.embed_text`` is what you embed or BM25-index. With
``prefix=True`` it is the section path, a blank line and the text, which is the "free
prefix" arm of the v4 study (header chunks + section-path prefix; confirmatory: header chunks
beat 512-character windows on Qasper and FreshStack; scope: English, 7B indexing LLM).

``window`` gives token windows with overlap (default 256/32, the study's F256t arm, which
beat header chunks on FreshStack code docs). ``chunk`` falls back to windows over the prose
of input with no Markdown structure (converted PDFs, plain text).

Token counts default to a dependency-free estimate (``ceil(chars / 4)``). Pass
``tokenizer="tiktoken"`` (``pip install "mdkeychunker[tokens]"``; cl100k_base, the study's
tokenizer), a tiktoken ``Encoding``, any object with ``encode``, or a ``str -> int`` function.

Offsets: ``start_char``/``end_char`` index the original input (CRLF kept). The source slice
contains every line of ``text``; ``text`` joins blocks with one blank line, and split code or
table pieces repeat the fence or header row, so the slice equals ``text`` only up to that.
"""
from __future__ import annotations

import bisect
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast

from .chunker import MarkdownChunker, _Block
from .models import Chunk

if TYPE_CHECKING:
    from .config import Config

__all__ = ["chunk", "chunk_file", "window", "HeaderlessWarning", "Tokenizer"]

Tokenizer = Union[None, str, Callable[[str], int], Any]
"""``None`` (chars/4 estimate), ``"tiktoken"`` / ``"tiktoken:<encoding>"``, a tiktoken
``Encoding``, an object with ``encode(text)``, or a function returning a token count."""

Headerless = Literal["auto", "window", "structure"]

_WORD = re.compile(r"\S+\s*")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# Prose split points, coarsest first: blank lines, line breaks, sentence ends, whitespace.
_PROSE_SEPS = (re.compile(r"\n\s*\n"), re.compile(r"\n"),
               re.compile(r"(?<=[.!?。！？])\s+"), re.compile(r"\s+"))
_CODE_SEPS = (re.compile(r"\n"), re.compile(r"\s+"))
_ATOMIC = {"code", "table"}  # never windowed, even in headerless mode
_STRUCTURE = {"code", "table", "list"}  # Markdown structure worth cutting at
_BINARY_SUFFIXES = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".epub"}


class HeaderlessWarning(UserWarning):
    """``chunk`` found too few headers and used token windows instead."""


# --------------------------------------------------------------------------- #
# Tokenizers                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class _Unit:
    start: int
    end: int
    cost: float


class _Tok:
    """Token counting plus a split of text into contiguous costed units (for windows)."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)

    def units(self, text: str) -> list[_Unit]:
        return [_Unit(m.start(), m.end(), self._cost(m.group())) for m in _WORD.finditer(text)]

    def _cost(self, piece: str) -> float:
        return len(piece) / 4


class _CountTok(_Tok):
    def __init__(self, fn: Callable[[str], int]):
        self.fn = fn

    def count(self, text: str) -> int:
        return int(self.fn(text)) if text else 0

    def _cost(self, piece: str) -> float:
        return float(self.fn(piece))


class _TiktokenTok(_Tok):
    """Exact token windows: units are tokens, with offsets from ``decode_with_offsets``."""

    def __init__(self, enc: Any):
        self.enc = enc

    def count(self, text: str) -> int:
        return len(self.enc.encode(text, disallowed_special=()))

    def units(self, text: str) -> list[_Unit]:
        toks = self.enc.encode(text, disallowed_special=())
        decoded, offsets = self.enc.decode_with_offsets(toks)
        if decoded != text:  # cannot happen for valid str input; fall back to word units
            return super().units(text)
        units: list[_Unit] = []
        for off in offsets:  # tokens that split one character share its start offset
            if units and units[-1].start == off:
                units[-1].cost += 1
            else:
                if units:
                    units[-1].end = off
                units.append(_Unit(off, len(text), 1.0))
        return units


def _resolve_tokenizer(tokenizer: Tokenizer) -> _Tok:
    if tokenizer is None or tokenizer == "estimate":
        return _Tok()
    if isinstance(tokenizer, str):
        name, _, encoding = tokenizer.partition(":")
        if name != "tiktoken":
            raise ValueError(f"unknown tokenizer {tokenizer!r}; use None, 'tiktoken' or "
                             "'tiktoken:<encoding>', an encoder object or a count function")
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                'tokenizer="tiktoken" needs: pip install "mdkeychunker[tokens]"') from e
        return _TiktokenTok(tiktoken.get_encoding(encoding or "cl100k_base"))
    if hasattr(tokenizer, "decode_with_offsets") and hasattr(tokenizer, "encode"):
        return _TiktokenTok(tokenizer)
    if hasattr(tokenizer, "encode"):
        return _CountTok(lambda s: len(tokenizer.encode(s)))
    if callable(tokenizer):
        return _CountTok(tokenizer)
    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__name__}")


# --------------------------------------------------------------------------- #
# Windows                                                                     #
# --------------------------------------------------------------------------- #

def _window_spans(text: str, tok: _Tok, size: int, overlap: int) -> list[tuple[int, int]]:
    units: list[_Unit] = []
    for u in tok.units(text):
        if u.cost <= size:
            units.append(u)
            continue
        # A unit over the window (a very long "word"): cut it into equal character pieces
        n = math.ceil(u.cost / size)
        step = math.ceil((u.end - u.start) / n)
        for s in range(u.start, u.end, step):
            e = min(s + step, u.end)
            units.append(_Unit(s, e, u.cost * (e - s) / (u.end - u.start)))
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        j, cost = i, 0.0
        while j < len(units) and (j == i or cost + units[j].cost <= size):
            cost += units[j].cost
            j += 1
        # Unit costs are additive only approximately (BPE merges across a cut): re-check
        while j - 1 > i and tok.count(text[units[i].start:units[j - 1].end].strip()) > size:
            j -= 1
        spans.append((units[i].start, units[j - 1].end))
        if j >= len(units):
            break
        k, back = j, 0.0  # step back by up to `overlap` tokens, always moving forward
        while k - 1 > i and back + units[k - 1].cost <= overlap:
            k -= 1
            back += units[k].cost
        i = k
    return [s for s in (_trim(text, a, b) for a, b in spans) if s]


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    seg = text[start:end]
    lead = len(seg) - len(seg.lstrip())
    body = seg.strip()
    return (start + lead, start + lead + len(body)) if body else None


def window(text: str, size: int = 256, overlap: int = 32, *, doc_id: str = "",
           tokenizer: Tokenizer = None, metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Split ``text`` into overlapping windows of at most ``size`` tokens.

    Windows break at whitespace (or at token boundaries with a tiktoken tokenizer) and each
    ``text`` is an exact slice ``input[start_char:end_char]``, trimmed of outer whitespace.
    256/32 is the study's token-window arm, which beat header chunks on FreshStack
    (confirmatory, -2.9 points; code docs, English).

    >>> words = lambda s: len(s.split())
    >>> cs = window("one two three four five six", size=2, overlap=1, tokenizer=words)
    >>> [c.text for c in cs]
    ['one two', 'two three', 'three four', 'four five', 'five six']
    >>> cs[1].start_char, cs[1].end_char
    (4, 13)
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    tok = _resolve_tokenizer(tokenizer)
    lines = _line_starts(text, "\n")
    chunks = [Chunk(text=text[s:e], embed_text=text[s:e], content_types=["window"], doc_id=doc_id,
                    start_char=s, end_char=e, start_line=bisect.bisect_right(lines, s) - 1,
                    end_line=bisect.bisect_right(lines, e) - 1, metadata=dict(metadata or {}))
              for s, e in _window_spans(text, tok, size, overlap)]
    return _finalize(chunks, tok)


# --------------------------------------------------------------------------- #
# Oversized splitting                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class _Piece:
    """A unit of chunk text with its span in the normalised document."""
    text: str
    start: int
    end: int
    types: set[str] = field(default_factory=set)
    path: list[str] = field(default_factory=list)
    oversized: bool = False
    header: bool = False  # headers travel with the content that follows them


def _split_text(text: str, lo: int, hi: int, budget: int, tok: _Tok,
                seps: tuple[re.Pattern[str], ...]) -> list[tuple[int, int]]:
    """Spans of ``text[lo:hi]`` each within ``budget`` tokens, cut at the coarsest separator."""
    span = _trim(text, lo, hi)
    if span is None:
        return []
    lo, hi = span
    if tok.count(text[lo:hi]) <= budget:
        return [(lo, hi)]
    for n, sep in enumerate(seps):
        cuts = [m.end() for m in sep.finditer(text, lo, hi) if lo < m.start() and m.end() < hi]
        if not cuts:
            continue
        parts: list[tuple[int, int]] = []
        for a, b in zip([lo, *cuts], [*cuts, hi]):
            parts.extend(_split_text(text, a, b, budget, tok, seps[n + 1:]))
        return _pack_adjacent(text, parts, budget, tok)
    return _hard_cut(text, lo, hi, budget, tok)


def _pack_adjacent(text: str, spans: list[tuple[int, int]], budget: int,
                   tok: _Tok) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and tok.count(text[out[-1][0]:e]) <= budget:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _hard_cut(text: str, lo: int, hi: int, budget: int, tok: _Tok) -> list[tuple[int, int]]:
    """Longest prefixes within budget (binary search); the last resort for unbroken text."""
    out: list[tuple[int, int]] = []
    while lo < hi:
        a, b = lo + 1, hi
        while a < b:
            mid = (a + b + 1) // 2
            if tok.count(text[lo:mid]) <= budget:
                a = mid
            else:
                b = mid - 1
        out.append((lo, a))
        lo = a
    return out


def _split_code(b: _Block, start: int, budget: int, tok: _Tok) -> list[_Piece] | None:
    lines = b.content.split("\n")
    m = _FENCE.match(lines[0])
    if m is None or len(lines) < 2:
        return None
    closed = re.match(
        rf"^ {{0,3}}{re.escape(m.group(1)[0])}{{{len(m.group(1))},}}[ \t]*$", lines[-1])
    opener, closer = lines[0], (lines[-1] if closed else m.group(1))
    body_lo = len(lines[0]) + 1
    body_hi = len(b.content) - (len(lines[-1]) + 1 if closed else 0)
    room = budget - tok.count(opener + "\n\n" + closer)
    if body_lo >= body_hi or room < max(8, budget // 4):
        return None
    spans = _split_text(b.content, body_lo, body_hi, room, tok, _CODE_SEPS)
    pieces = []
    for i, (s, e) in enumerate(spans):
        s0 = 0 if i == 0 else s
        e0 = len(b.content) if i == len(spans) - 1 else e
        pieces.append(_Piece(f"{opener}\n{b.content[s:e]}\n{closer}", start + s0, start + e0,
                             {b.type}, b.section_path, True))
    return pieces


def _split_table(b: _Block, start: int, budget: int, tok: _Tok) -> list[_Piece] | None:
    lines = b.content.split("\n")
    if len(lines) < 3:
        return None
    head = lines[0] + "\n" + lines[1]
    room = budget - tok.count(head + "\n")
    if room < max(8, budget // 4):
        return None
    offs, pos = [], len(head) + 1
    for line in lines[2:]:
        offs.append((pos, pos + len(line)))
        pos += len(line) + 1
    groups: list[list[tuple[int, int]]] = []
    for s, e in offs:
        if groups and tok.count(b.content[groups[-1][0][0]:e]) <= room:
            groups[-1].append((s, e))
        else:
            groups.append([(s, e)])
    pieces: list[_Piece] = []
    for g in groups:
        s, e = g[0][0], g[-1][1]
        # A single row over the cap is split as text; every piece keeps the header row
        spans = ([(s, e)] if tok.count(b.content[s:e]) <= room
                 else _split_text(b.content, s, e, room, tok, _PROSE_SEPS))
        for rs, re_ in spans:
            pieces.append(_Piece(f"{head}\n{b.content[rs:re_]}", start + (0 if not pieces else rs),
                                 start + re_, {b.type}, b.section_path, True))
    return pieces


def _split_block(b: _Block, start: int, budget: int, tok: _Tok) -> list[_Piece]:
    pieces = None
    if b.type == "code":
        pieces = _split_code(b, start, budget, tok)
    elif b.type == "table":
        pieces = _split_table(b, start, budget, tok)
    if pieces is None:
        seps = _CODE_SEPS if b.type == "code" else _PROSE_SEPS
        pieces = [_Piece(b.content[s:e], start + s, start + e, {b.type}, b.section_path, True)
                  for s, e in _split_text(b.content, 0, len(b.content), budget, tok, seps)]
    return pieces


def _cap(blocks: list[_Block], starts: list[int], room: Callable[[list[str]], int],
         tok: _Tok) -> list[list[_Piece]]:
    """Re-pack one over-cap chunk's blocks into groups that fit.

    ``room(path)`` is the text budget of a group whose section path is ``path`` (the cap
    minus that group's prefix). Each group is sized by its own path only, so content after
    it never moves its boundaries (ids stay stable when a document grows).
    """
    def fits(group: list[_Piece], text: str) -> bool:
        return tok.count(text) <= room(_body_path([(p.path, p.header) for p in group]))

    units: list[_Piece] = []
    for b in blocks:
        s = starts[b.start_line]
        cap = room(b.section_path)
        if b.type == "header" and tok.count(b.content) <= cap:
            units.append(_Piece(b.content, s, s + len(b.content), {"header"}, b.section_path,
                                header=True))
        elif tok.count(b.content) <= cap:
            units.append(_Piece(b.content, s, s + len(b.content), {b.type}, b.section_path))
        else:
            # Leave room for the headers that will travel with the first piece
            lead = tok.count("\n\n".join(u.text for u in _trailing_headers(units)) + "\n\n")
            budget = cap - lead if lead <= cap // 2 else cap
            units.extend(_split_block(b, s, budget, tok))

    groups: list[list[_Piece]] = []
    cur: list[_Piece] = []
    cur_text = ""
    for u in units:
        if cur and not fits(cur + [u], cur_text + "\n\n" + u.text):
            tail = _trailing_headers(cur)  # headers travel with the content that follows
            body = cur[:len(cur) - len(tail)]
            if body:
                groups.append(body)
                cur = tail
                cur_text = "\n\n".join(p.text for p in cur)
            if cur and not fits(cur + [u], cur_text + "\n\n" + u.text):
                groups.append(cur)  # a header run too long to travel: emitted on its own
                cur, cur_text = [], ""
        cur.append(u)
        cur_text = cur_text + "\n\n" + u.text if cur_text else u.text
    if cur:
        last = "\n\n".join(p.text for p in groups[-1]) if groups else ""
        if all(p.header for p in cur) and groups \
                and fits(groups[-1] + cur, last + "\n\n" + cur_text):
            groups[-1].extend(cur)  # trailing headers with no content of their own
        else:
            groups.append(cur)
    return groups


def _trailing_headers(units: list[_Piece]) -> list[_Piece]:
    n = len(units)
    while n and units[n - 1].header:
        n -= 1
    return units[n:]


# --------------------------------------------------------------------------- #
# chunk                                                                       #
# --------------------------------------------------------------------------- #

def _line_starts(text: str, newline: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer(newline, text)]


def _body_path(parts: list[tuple[list[str], bool]]) -> list[str]:
    """Header stack at the first body (non-header) part; a header-only chunk takes its last."""
    return next((path for path, is_header in parts if not is_header), parts[-1][0])


def _embed_text(path: list[str], text: str, prefix: bool) -> str:
    return f"{' > '.join(path)}\n\n{text}" if prefix and path else text


def _finalize(chunks: list[Chunk], tok: _Tok) -> list[Chunk]:
    for i, c in enumerate(chunks):
        c.position_index = i
        c.token_count = tok.count(c.embed_text or c.text)
        c.stable_id()
    for prev, nxt in zip(chunks, chunks[1:]):
        prev.next_chunk_id, nxt.previous_chunk_id = nxt.chunk_id, prev.chunk_id
    return chunks


def chunk(markdown: str, doc_id: str = "", *, max_tokens: int | None = 512, prefix: bool = True,
          tokenizer: Tokenizer = None, headerless: Headerless = "auto", min_chars: int = 100,
          max_chars: int = 1500, metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Chunk a Markdown document at its headers, capped at ``max_tokens``.

    Args:
        markdown: the document. CRLF and CR line endings are accepted; offsets refer to it.
        doc_id: document id, copied to every chunk and hashed into ``chunk_id``.
        max_tokens: hard cap on each chunk's ``embed_text`` (what gets embedded), so the
            prefix counts; ``token_count`` is that count. ``None``: no cap, the 0.3 / study
            chunks exactly. Over-cap chunks are re-packed at block boundaries; an atomic
            block over the cap is split inside and its pieces get ``oversized=True``. A run
            of headers too long to travel with its content (over half the cap) becomes a
            chunk of its own rather than break the cap.
        prefix: ``embed_text`` = section path + blank line + text (the study's free prefix).
        tokenizer: see :data:`Tokenizer`; default estimate ``ceil(chars / 4)``.
        headerless: ``"auto"`` switches to 256-token windows (overlap 32) with a
            :class:`HeaderlessWarning` when the document has no Markdown structure to cut
            at (no headers, or one header and no code, tables or lists) and is longer than
            one window; ``"window"`` always does; ``"structure"`` never does. Windows run
            over prose only: fenced code and tables stay whole, or are split row-/line-aware.
        min_chars, max_chars: the structural chunker's soft sizes (0.3 / study defaults).
        metadata: copied (shallowly) into every chunk's ``metadata``.

    ``section_path`` is the header stack at the chunk's first body line, outermost first;
    with ``prefix`` it is joined with ``" > "`` for ``embed_text``. ``section_title`` keeps
    its 0.3 value, which for a chunk merged across headers can name a later section. The v4
    study's prefix used ``section_title``, so ``embed_text`` differs from the study's indexed
    text for such chunks (5 of 184 on the repo's own Markdown files); the exact study
    rendering is ``section_title + "\n\n" + text`` with ``max_tokens=None``. ``chunk_id`` is
    content-stable (``Chunk.stable_id``: doc_id, start offset, text): editing or appending
    text leaves the ids of all chunks before the change unchanged.

    >>> cs = chunk("# Guide\\n\\n## Install\\n\\nRun pip install.", doc_id="g")
    >>> cs[0].section_path, cs[0].text
    (['Guide', 'Install'], '# Guide\\n\\n## Install\\n\\nRun pip install.')
    >>> print(cs[0].embed_text.split("\\n")[0])
    Guide > Install
    """
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None")
    if headerless not in ("auto", "window", "structure"):
        raise ValueError("headerless must be 'auto', 'window' or 'structure'")
    tok = _resolve_tokenizer(tokenizer)
    if not markdown.strip():
        return []

    sizes = SimpleNamespace(min_chunk_size=min_chars, max_chunk_size=max_chars)
    mc = MarkdownChunker(cast("Config", sizes))
    blocks = mc._parse_blocks(markdown)
    norm = markdown.replace("\r\n", "\n").replace("\r", "\n")
    n_starts = _line_starts(norm, "\n")
    o_starts = _line_starts(markdown, r"\r\n|\r|\n")

    def orig(n_off: int) -> int:
        i = bisect.bisect_right(n_starts, n_off) - 1
        return o_starts[i] + (n_off - n_starts[i])

    def span(b: _Block) -> tuple[int, int]:
        s = n_starts[b.start_line]
        return s, s + len(b.content)

    def pre_tokens(path: list[str]) -> int:
        return tok.count(_embed_text(path, "", True)) if prefix and path else 0

    def use_prefix(path: list[str]) -> bool:
        # A prefix over half the cap (an absurdly long title) is left out instead
        return max_tokens is None or pre_tokens(path) <= max_tokens // 2

    def room(path: list[str]) -> int | None:
        """Text tokens left under the cap once the prefix is added (the cap is on embed_text)."""
        if max_tokens is None:
            return None
        return max_tokens - (pre_tokens(path) if use_prefix(path) else 0)

    def make(text: str, start: int, end: int, path: list[str], title: str, types: set[str],
             oversized: bool) -> Chunk:
        return Chunk(text=text, section_title=title, section_path=list(path),
                     content_types=sorted(types), doc_id=doc_id,
                     start_line=bisect.bisect_right(n_starts, start) - 1,
                     end_line=bisect.bisect_right(n_starts, end) - 1,
                     start_char=orig(start), end_char=orig(end), oversized=oversized,
                     embed_text=_embed_text(path, text, prefix and use_prefix(path)),
                     metadata=dict(metadata or {}))

    def capped(group_blocks: list[_Block]) -> list[Chunk]:
        out = []
        for g in _cap(group_blocks, n_starts, lambda path: cast(int, room(path)), tok):
            path = _body_path([(p.path, p.header) for p in g])
            out.append(make("\n\n".join(p.text for p in g), g[0].start, g[-1].end, path,
                            " > ".join(path), set().union(*(p.types for p in g)),
                            any(p.oversized for p in g)))
        return out

    n_headers = sum(b.type == "header" for b in blocks)
    structured = any(b.type in _STRUCTURE for b in blocks)
    size = min(256, max_tokens or 256)
    plain = n_headers == 0 or (n_headers < 2 and not structured)
    if headerless == "window" or (headerless == "auto" and plain and tok.count(markdown) > size):
        if headerless == "auto":
            warnings.warn(f"{doc_id or 'document'}: no Markdown headers or structure to cut at; "
                          f"using {size}-token windows (headerless='structure' keeps paragraphs)",
                          HeaderlessWarning, stacklevel=2)
        return _finalize(_windowed(blocks, norm, span, room, make, tok, size), tok)

    out: list[Chunk] = []
    bi = 0
    for legacy in mc._group_blocks(blocks):
        while blocks[bi].start_line < legacy.start_line:
            bi += 1
        mine = []
        while bi < len(blocks) and blocks[bi].end_line <= legacy.end_line:
            mine.append(blocks[bi])
            bi += 1
        # section_path: the header stack at the chunk's first line (RESEARCH.md sec. 4, inv. 5);
        # section_title keeps the 0.3 value, which can name a later section of a merged chunk
        path = _body_path([(b.section_path, b.type == "header") for b in mine])
        fits = room(path)
        if fits is None or tok.count(legacy.text) <= fits:
            out.append(make(legacy.text, span(mine[0])[0], span(mine[-1])[1], path,
                            legacy.section_title, set(legacy.content_types), False))
        else:
            out.extend(capped(mine))
    return _finalize(out, tok)


def _windowed(blocks: list[_Block], norm: str, span: Callable[[_Block], tuple[int, int]],
              room: Callable[[list[str]], int | None], make: Callable[..., Chunk], tok: _Tok,
              size: int) -> list[Chunk]:
    """Headerless mode: token windows over runs of prose; code and tables stay atomic."""
    out: list[Chunk] = []
    run: list[_Block] = []

    def flush_run() -> None:
        if not run:
            return
        lo, hi = span(run[0])[0], span(run[-1])[1]
        starts = [span(b)[0] for b in run]
        fits = [r for r in (room(b.section_path) for b in run) if r is not None]
        for s, e in _window_spans(norm[lo:hi], tok, min([size, *fits]), min(32, size // 8)):
            b = run[bisect.bisect_right(starts, lo + s) - 1]
            out.append(make(norm[lo + s:lo + e], lo + s, lo + e, b.section_path,
                            " > ".join(b.section_path), {"window"}, False))
        run.clear()

    for b in blocks:
        if b.type not in _ATOMIC:
            run.append(b)
            continue
        flush_run()
        s, e = span(b)
        fits = room(b.section_path)
        if fits is None or tok.count(b.content) <= fits:
            out.append(make(b.content, s, e, b.section_path, " > ".join(b.section_path),
                            {b.type}, False))
        else:
            out.extend(make(p.text, p.start, p.end, p.path, " > ".join(p.path), p.types, True)
                       for p in _split_block(b, s, fits, tok))
    flush_run()
    return out


def chunk_file(path: str | Path, doc_id: str | None = None, *, encoding: str = "utf-8",
               **kwargs: Any) -> list[Chunk]:
    """Read a Markdown or plain-text file and :func:`chunk` it.

    ``doc_id`` defaults to the path as given (POSIX form). Line endings are kept, so offsets
    index the file's decoded characters. Binary formats are rejected:
    convert PDF/DOCX to Markdown first (Docling is recommended for PDFs; MarkItDown works
    for DOCX/HTML but is weak on PDFs). Other keyword arguments go to :func:`chunk`.
    """
    p = Path(path)
    if p.suffix.lower() in _BINARY_SUFFIXES:
        raise ValueError(f"{p.name}: convert {p.suffix} to Markdown first (e.g. Docling for "
                         "PDFs, MarkItDown for DOCX/HTML), then call mk.chunk(markdown)")
    text = p.read_bytes().decode(encoding)
    return chunk(text, p.as_posix() if doc_id is None else doc_id, **kwargs)
