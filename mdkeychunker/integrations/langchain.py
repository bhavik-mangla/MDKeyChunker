"""LangChain ``TextSplitter`` backed by ``mk.chunk`` (``pip install "mdkeychunker[langchain]"``).

>>> from mdkeychunker.integrations.langchain import MDKeyChunkerSplitter  # doctest: +SKIP
>>> docs = MDKeyChunkerSplitter(max_tokens=512).split_documents(loaded_docs)  # doctest: +SKIP
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('The LangChain adapter needs: pip install "mdkeychunker[langchain]"') from e

from ..chunking import Tokenizer
from ._common import Content, check_content, chunk_for, content_of, mk_metadata

__all__ = ["MDKeyChunkerSplitter"]


def _doc_id(text: str, metadata: dict[str, Any], explicit: Any = None) -> str:
    if explicit:
        return str(explicit)
    if metadata.get("source"):
        return str(metadata["source"])
    return hashlib.sha256(text.encode()).hexdigest()[:16]  # stable without an id or source


class MDKeyChunkerSplitter(TextSplitter):
    """Header-first chunks capped at ``max_tokens``, as LangChain ``Document`` objects.

    ``page_content`` is ``Chunk.embed_text`` by default: the section path, a blank line and
    the text, which is what the v4 study indexed as its free prefix (confirmatory: header
    chunks beat 512-character windows on Qasper and FreshStack). Pass ``content="text"``
    for the raw chunk text; the path is always in ``metadata["section"]``.

    Each document gets ``id`` = the content-stable ``chunk_id`` and metadata ``chunk_id``,
    ``section``, ``start_index`` / ``end_index`` (offsets into the source text, LangChain's
    ``add_start_index`` convention), ``token_count`` and ``oversized``, on top of the source
    document's metadata. Other keyword arguments go to :func:`mk.chunk`.
    """

    def __init__(self, max_tokens: int | None = 512, *, prefix: bool = True,
                 content: Content = "embed_text", tokenizer: Tokenizer = None,
                 headerless: str = "auto", **chunk_kwargs: Any) -> None:
        super().__init__(chunk_size=max_tokens or 2**31 - 1, chunk_overlap=0,
                         strip_whitespace=False)
        self.max_tokens, self.prefix = max_tokens, prefix
        self.content = check_content(content)
        self.tokenizer, self.headerless, self.chunk_kwargs = tokenizer, headerless, chunk_kwargs

    def _chunks(self, text: str, doc_id: str) -> list[Any]:
        return chunk_for(text, doc_id, max_tokens=self.max_tokens, prefix=self.prefix,
                         tokenizer=self.tokenizer, headerless=self.headerless,
                         chunk_kwargs=self.chunk_kwargs)

    def split_text(self, text: str) -> list[str]:
        return [content_of(c, self.content) for c in self._chunks(text, _doc_id(text, {}))]

    def _documents(self, text: str, metadata: dict[str, Any], doc_id: str) -> list[Document]:
        return [Document(page_content=content_of(c, self.content), id=c.chunk_id,
                         metadata={**metadata, **mk_metadata(c)})
                for c in self._chunks(text, doc_id)]

    def create_documents(self, texts: list[str],
                         metadatas: list[dict[Any, Any]] | None = None) -> list[Document]:
        metas = metadatas or [{} for _ in texts]
        return [d for text, meta in zip(texts, metas)
                for d in self._documents(text, dict(meta), _doc_id(text, dict(meta)))]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        return [d for doc in documents
                for d in self._documents(doc.page_content, dict(doc.metadata),
                                         _doc_id(doc.page_content, dict(doc.metadata),
                                                 getattr(doc, "id", None)))]
