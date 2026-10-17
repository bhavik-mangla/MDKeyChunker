"""Chunk-agnostic retrieval metrics: gold evidence found within a token budget.

Ported from the v4 study harness (``benchmarks/chunking_study/evaluate.py`` and
``evaluate_corpus.py``) so that numbers computed here match the paper's when the same
tokenizer is used. Gold evidence is defined on the source text, not on chunk ids, so
different chunkers can be compared on the same questions.

Rules (identical to the study):

- Retrieved chunks are taken in rank order and their raw ``text`` (not ``embed_text``)
  fills ``budget_tokens``; the chunk that crosses the budget is cut there and its kept
  part counts. Chunks stay separate units, so word n-grams never straddle two chunks.
- A text span is covered when at least half of its word 5-grams occur in the retrieved
  units. A span of fewer than five words must appear as a contiguous word sequence inside
  one unit. Words are lower-cased ASCII alphanumeric runs (the study is English-only).

Additions for the library (not used in the study):

- ``GoldSpan`` with ``start``/``end`` offsets is covered when at least half of its
  characters fall inside the kept part of retrieved chunks of the same ``doc_id``. This
  needs chunks with offsets (``mk.chunk`` / ``mk.window``).
- The same chunk retrieved twice counts once and does not use the budget twice.
- The default tokenizer is dependency-free (one token per four characters, rounded up).
  The study used tiktoken ``cl100k_base``; pass ``tokenizer="cl100k_base"`` (needs
  ``pip install "mdkeychunker[tokens]"``) to reproduce its numbers.

Example:
    >>> from mdkeychunker.models import Chunk, GoldSpan
    >>> c1 = Chunk(text="The cat sat on the mat all day long.", chunk_id="c1")
    >>> c2 = Chunk(text="Dogs bark at night.", chunk_id="c2")
    >>> evidence_recall([c2, c1], ["the cat sat on the mat"], budget_tokens=512)
    1.0
    >>> evidence_recall([c2, c1], ["the cat sat on the mat"], budget_tokens=5)
    0.0
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Union, runtime_checkable

from .models import Chunk, GoldSpan, Hit, RetrievedItem

__all__ = [
    "AnyTokenizer",
    "CharTokenizer",
    "GoldLike",
    "Tokenizer",
    "evidence_precision",
    "evidence_recall",
    "get_tokenizer",
    "to_gold_spans",
]

NGRAM = 5
THRESHOLD = 0.5
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- tokenizers

@runtime_checkable
class Tokenizer(Protocol):
    """What the budget needs: ``encode`` text to token ids and ``decode`` a prefix back.

    A tiktoken ``Encoding`` satisfies it, as does any object with these two methods.
    """

    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


class CharTokenizer:
    """Dependency-free estimate: one token per ``chars_per_token`` characters (rounded up).

    Stateless (safe to share across threads). It has ``count``/``truncate`` instead of
    ``encode``/``decode``; the budget logic uses those directly.

    Example:
        >>> t = CharTokenizer()
        >>> t.count("abcdefghij"), t.truncate("abcdefghij", 2)
        (3, 'abcdefgh')
    """

    def __init__(self, chars_per_token: int = 4) -> None:
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be >= 1")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token)

    def truncate(self, text: str, n_tokens: int) -> str:
        """The first ``n_tokens`` tokens of ``text``."""
        return text[: max(n_tokens, 0) * self.chars_per_token]


AnyTokenizer = Union[Tokenizer, CharTokenizer]


def _cut(tok: AnyTokenizer, text: str, n_tokens: int) -> tuple[str, int]:
    """Kept text and tokens used when at most ``n_tokens`` tokens of ``text`` fit."""
    if isinstance(tok, CharTokenizer):
        used = min(tok.count(text), n_tokens)
        return tok.truncate(text, used), used
    ids = list(tok.encode(text))[:n_tokens]
    return tok.decode(ids), len(ids)


def get_tokenizer(tokenizer: Tokenizer | CharTokenizer | str | None = None) -> AnyTokenizer:
    """Resolve a tokenizer spec, accepting the same strings as ``mk.chunk``.

    ``None`` or ``"estimate"``: the chars/4 estimate. ``"tiktoken"``: tiktoken
    ``cl100k_base`` (the study's). ``"tiktoken:<encoding>"`` or a bare ``"<encoding>"``
    (e.g. ``"cl100k_base"``): that tiktoken encoding. Or a tokenizer object.

    Example:
        >>> isinstance(get_tokenizer(None), CharTokenizer)
        True
        >>> isinstance(get_tokenizer("estimate"), CharTokenizer)
        True
    """
    if tokenizer is None or tokenizer == "estimate":
        return CharTokenizer()
    if isinstance(tokenizer, str):
        name, sep, encoding = tokenizer.partition(":")
        if name == "tiktoken":
            encoding = encoding or "cl100k_base"
        elif sep:
            raise ValueError(f"unknown tokenizer {tokenizer!r}; use None, 'estimate', "
                             "'tiktoken', 'tiktoken:<encoding>' or '<encoding>'")
        else:
            encoding = tokenizer
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(f'tokenizer={tokenizer!r} needs tiktoken: '
                              'pip install "mdkeychunker[tokens]"') from e
        enc: Tokenizer = tiktoken.get_encoding(encoding)
        return enc
    if isinstance(tokenizer, CharTokenizer):
        return tokenizer
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be None, a tiktoken encoding name, or have encode/decode")
    return tokenizer


# --------------------------------------------------------------------------- gold spans

GoldLike = Union[GoldSpan, str, Mapping[str, Any], Sequence[Any]]


def to_gold_spans(gold: Iterable[GoldLike]) -> list[GoldSpan]:
    """Coerce gold evidence to ``GoldSpan`` objects.

    Accepts a ``GoldSpan``, a string (span text), a dict with ``text`` or ``start``/``end``
    (and optional ``doc_id``), or a ``(doc_id, start, end)`` tuple.

    Example:
        >>> to_gold_spans(["some text", ("d1", 0, 10)])[1]
        GoldSpan(doc_id='d1', text='', start=0, end=10)
    """
    out = []
    for g in gold:
        if isinstance(g, GoldSpan):
            span = g
        elif isinstance(g, str):
            span = GoldSpan(text=g)
        elif isinstance(g, Mapping):
            span = GoldSpan(doc_id=str(g.get("doc_id", "")), text=str(g.get("text", "")),
                            start=g.get("start"), end=g.get("end"))
        elif isinstance(g, Sequence) and len(g) == 3:
            span = GoldSpan(doc_id=str(g[0]), start=int(g[1]), end=int(g[2]))
        else:
            raise TypeError(f"cannot read gold evidence from {g!r}")
        if (span.start is None) != (span.end is None):
            raise ValueError(f"gold span needs both start and end: {span!r}")
        if span.start is not None and span.end is not None:
            if span.end <= span.start:
                raise ValueError(f"gold span is empty: {span!r}")
        elif not span.text.strip():
            raise ValueError(f"gold span has neither text nor offsets: {span!r}")
        out.append(span)
    return out


def _is_offset(g: GoldSpan) -> bool:
    return g.start is not None


# --------------------------------------------------------------------------- matching (ported)

def words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def ngrams(ws: Sequence[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)} if len(ws) >= n else {tuple(ws)}


def _unit_grams(units: Sequence[Sequence[str]]) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for u in units:
        if u:
            out |= ngrams(u)
    return out


def _covered_text(evidence: str, units: Sequence[Sequence[str]],
                  grams: set[tuple[str, ...]]) -> bool:
    ws = words(evidence)
    if len(ws) < NGRAM:  # short evidence must appear as a contiguous word sequence in one chunk
        n = len(ws)
        return bool(ws) and any(list(u[i:i + n]) == ws
                                for u in units for i in range(len(u) - n + 1))
    eg = ngrams(ws)
    return sum(g in grams for g in eg) / len(eg) >= THRESHOLD


# --------------------------------------------------------------------------- budget

@dataclass
class _Unit:
    chunk: Chunk
    kept: str            # the part of chunk.text inside the budget
    words: list[str]
    start: int           # kept character range in the source document (offset gold only)
    end: int


def _resolve(item: RetrievedItem, lookup: Mapping[str, Chunk] | None) -> Chunk:
    if isinstance(item, Hit):
        return item.chunk
    if isinstance(item, Chunk):
        return item
    if isinstance(item, str):
        if lookup is None:
            raise ValueError("results are chunk ids: pass chunks= so they can be looked up")
        try:
            return lookup[item]
        except KeyError:
            raise KeyError(f"retrieved chunk id {item!r} is not in chunks") from None
    raise TypeError(f"cannot read a retrieved item of type {type(item).__name__}")


def _lookup(chunks: Mapping[str, Chunk] | Iterable[Chunk] | None) -> Mapping[str, Chunk] | None:
    if chunks is None:
        return None
    if isinstance(chunks, Mapping):
        return {str(k): v for k, v in chunks.items()}
    out: dict[str, Chunk] = {}
    for c in chunks:
        if not c.chunk_id:
            raise ValueError("every chunk passed as chunks= needs a chunk_id")
        out[c.chunk_id] = c
    return out


def _budget_units(results: Sequence[RetrievedItem], budget_tokens: int | None, k: int | None,
                  tok: AnyTokenizer, lookup: Mapping[str, Chunk] | None) -> list[_Unit]:
    if budget_tokens is not None and budget_tokens < 0:
        raise ValueError("budget_tokens must be >= 0")
    units: list[_Unit] = []
    seen: set[object] = set()
    used = 0
    for item in (results[:k] if k is not None else results):
        if budget_tokens is not None and used >= budget_tokens:
            break
        c = _resolve(item, lookup)
        ident: object = c.chunk_id or id(c)
        if ident in seen:
            continue
        seen.add(ident)
        kept = c.text
        if budget_tokens is not None:
            kept, n = _cut(tok, c.text, budget_tokens - used)
            used += n
        span = c.end_char - c.start_char
        frac = len(kept) / len(c.text) if c.text else 0.0
        units.append(_Unit(c, kept, words(kept), c.start_char, c.start_char + round(span * frac)))
    return units


def _same_doc(g: GoldSpan, c: Chunk) -> bool:
    return not g.doc_id or not c.doc_id or g.doc_id == c.doc_id


def _check_offsets(units: Sequence[_Unit]) -> None:
    for u in units:
        if u.chunk.text and u.chunk.end_char <= u.chunk.start_char:
            raise ValueError(
                "gold spans with start/end need chunks with character offsets (mk.chunk or "
                f"mk.window); chunk {u.chunk.chunk_id or u.chunk.text[:30]!r} has none")


def _covered_offset(g: GoldSpan, units: Sequence[_Unit]) -> bool:
    assert g.start is not None and g.end is not None
    ranges = sorted((max(u.start, g.start), min(u.end, g.end)) for u in units
                    if g.doc_id == u.chunk.doc_id and u.start < g.end and u.end > g.start)
    got, cur = 0, g.start
    for a, b in ranges:  # union of overlapping ranges
        a = max(a, cur)
        if b > a:
            got += b - a
            cur = b
    return got / (g.end - g.start) >= THRESHOLD


# --------------------------------------------------------------------------- public metrics

def evidence_recall(results: Sequence[RetrievedItem], gold: Iterable[GoldLike],
                    budget_tokens: int | None = 512, *, k: int | None = None,
                    tokenizer: Tokenizer | CharTokenizer | str | None = None,
                    chunks: Mapping[str, Chunk] | Iterable[Chunk] | None = None) -> float:
    """Fraction of gold spans found in the retrieved text within a token budget.

    The study's primary Qasper metric (recall within 512 tokens). See the module docstring
    for the matching rules.

    Args:
        results: ranked results for one question: ``Hit`` objects, ``Chunk`` objects or
            chunk ids (ids need ``chunks``).
        gold: the question's gold evidence (see ``to_gold_spans`` for accepted forms).
        budget_tokens: tokens of retrieved chunk text to read; ``None`` reads all results.
        k: read at most the first ``k`` results (applied before the budget).
        tokenizer: ``None`` (chars/4 estimate), a ``CharTokenizer``, a tiktoken encoding
            name such as ``"cl100k_base"``, or an object with ``encode``/``decode``.
        chunks: chunks (or an ``{chunk_id: Chunk}`` mapping) to resolve id results.

    Returns:
        A value in [0, 1].

    Note:
        For offset gold, the kept character range of a chunk cut by the budget is
        approximated proportionally: ``start_char + (end_char - start_char) * len(kept) /
        len(text)``. It is exact when ``text`` is the verbatim source slice.

    Raises:
        ValueError: if ``gold`` is empty, or offset gold is used with chunks that lack offsets.

    Example:
        >>> from mdkeychunker.models import Chunk
        >>> cs = [Chunk(text="alpha beta gamma delta epsilon", chunk_id="a"),
        ...       Chunk(text="zeta eta theta", chunk_id="b")]
        >>> evidence_recall(["b", "a"], ["alpha beta gamma delta epsilon", "eta"], chunks=cs)
        1.0
        >>> evidence_recall(["b", "a"], ["alpha beta gamma delta epsilon", "eta"], k=1, chunks=cs)
        0.5
    """
    spans = to_gold_spans(gold)
    if not spans:
        raise ValueError("evidence_recall needs at least one gold span")
    units = _budget_units(results, budget_tokens, k, get_tokenizer(tokenizer), _lookup(chunks))
    if any(_is_offset(g) for g in spans):
        _check_offsets(units)
    grams_by_doc: dict[str, tuple[list[list[str]], set[tuple[str, ...]]]] = {}
    hit = 0
    for g in spans:
        if _is_offset(g):
            hit += _covered_offset(g, units)
            continue
        if g.doc_id not in grams_by_doc:
            us = [u.words for u in units if _same_doc(g, u.chunk)]
            grams_by_doc[g.doc_id] = (us, _unit_grams(us))
        us, grams = grams_by_doc[g.doc_id]
        hit += _covered_text(g.text, us, grams)
    return hit / len(spans)


def evidence_precision(results: Sequence[RetrievedItem], gold: Iterable[GoldLike],
                       budget_tokens: int | None = 1024, *, k: int | None = None,
                       tokenizer: Tokenizer | CharTokenizer | str | None = None,
                       chunks: Mapping[str, Chunk] | Iterable[Chunk] | None = None) -> float:
    """Share of the retrieved text (within the budget) that is gold evidence.

    The study's primary FreshStack metric (gold-token precision within 1,024 tokens): the
    fraction of distinct word 5-grams of the kept retrieved text that occur in the gold
    spans. For offset gold it is the fraction of kept characters inside gold spans of the
    same document. Text and offset gold cannot be mixed in one call.

    Example:
        >>> from mdkeychunker.models import Chunk
        >>> gold = ["one two three four five six"]
        >>> evidence_precision([Chunk(text="one two three four five six")], gold)
        1.0
        >>> evidence_precision([Chunk(text="nine ten eleven twelve thirteen")], gold)
        0.0
    """
    spans = to_gold_spans(gold)
    if not spans:
        raise ValueError("evidence_precision needs at least one gold span")
    offset = [_is_offset(g) for g in spans]
    if any(offset) and not all(offset):
        raise ValueError("evidence_precision cannot mix text and offset gold spans")
    units = _budget_units(results, budget_tokens, k, get_tokenizer(tokenizer), _lookup(chunks))
    if all(offset):
        _check_offsets(units)
        kept = sum(u.end - u.start for u in units)
        inside = 0
        for u in units:
            pts = sorted((max(u.start, g.start or 0), min(u.end, g.end or 0)) for g in spans
                         if g.doc_id == u.chunk.doc_id
                         and (g.start or 0) < u.end and (g.end or 0) > u.start)
            cur = u.start
            for a, b in pts:
                a = max(a, cur)
                if b > a:
                    inside += b - a
                    cur = b
        return inside / kept if kept else 0.0
    gold_grams: set[tuple[str, ...]] = set()
    for g in spans:
        gold_grams |= ngrams(words(g.text))
    got = _unit_grams([u.words for u in units])  # every retrieved unit counts, as in the study
    return len(got & gold_grams) / len(got) if got else 0.0
