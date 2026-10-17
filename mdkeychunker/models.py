"""Data models for MDKeyChunker.

The 0.4 public contract: ``Chunk`` (with ``doc_id``, character offsets, ``section_path``
and ``embed_text``), ``Hit``, ``GoldSpan`` and the ``Retriever`` protocol. Fields used only
by the legacy ``Pipeline`` (keys, related keys, linked-list ids) are kept for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Protocol, Sequence, Union, runtime_checkable


@dataclass
class Chunk:
    text: str
    section_title: str = ""
    content_types: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    # Metadata (populated by enricher)
    chunk_id: str = ""
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)  # [{name, type}]
    questions: list[str] = field(default_factory=list)
    key: str = ""
    related_keys: list[str] = field(default_factory=list)
    position_index: int = 0
    previous_chunk_id: str = ""
    next_chunk_id: str = ""
    token_count: int = 0
    # 0.4 contract (populated by mk.chunk / mk.window / mk.enrich)
    doc_id: str = ""
    start_char: int = 0            # offset of the chunk's first character in the source document
    end_char: int = 0              # exclusive end offset
    section_path: list[str] = field(default_factory=list)  # header stack, outermost first
    embed_text: str = ""           # text to embed / BM25-index: section-path prefix (+ generated fields) + text
    oversized: bool = False        # an atomic block (code, table, paragraph) had to be split to respect max_tokens
    metadata: dict[str, Any] = field(default_factory=dict)  # user passthrough

    def stable_id(self) -> str:
        """Content-stable id: depends only on doc_id, start offset and text.

        Unlike ``generate_id`` (legacy), it does not change when enrichment or merging runs.
        """
        h = hashlib.sha256(f"{self.doc_id}\x00{self.start_char}\x00{self.text}".encode()).hexdigest()[:16]
        self.chunk_id = h
        return h

    def generate_id(self) -> str:
        h = hashlib.sha256(
            f"{self.section_title}:{self.key}:{self.position_index}:{self.text[:100]}".encode(
            )
        ).hexdigest()[:16]
        self.chunk_id = h
        return h

    def to_dict(self) -> dict:
        """Return all fields. Empty lists and zero values are preserved."""
        return dict(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, key={self.key!r}, "
            f"title={self.title!r}, tokens={self.token_count})"
        )


@dataclass
class Hit:
    """One retrieved chunk with its score and, for fused retrievers, the rank in each input list."""
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass
class GoldSpan:
    """Gold evidence, defined on the source text independently of any chunking.

    Give either ``text`` (matched inside chunk text after whitespace normalisation) or the
    character offsets ``start``/``end`` within document ``doc_id``.
    """
    doc_id: str = ""
    text: str = ""
    start: int | None = None
    end: int | None = None


RetrievedItem = Union[Hit, Chunk, str]  # a Hit, a Chunk, or a chunk_id


@runtime_checkable
class Retriever(Protocol):
    """Anything that maps a query to ranked results: ``search(query, k) -> list[...]``."""

    def search(self, query: str, k: int) -> Sequence[RetrievedItem]: ...


SearchFn = Callable[[str, int], Sequence[RetrievedItem]]  # bring-your-own retriever as a plain function
