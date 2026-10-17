"""Shared by the framework adapters: chunk a text and describe each chunk as flat metadata."""
from __future__ import annotations

from typing import Any, Literal

from ..chunking import Tokenizer, chunk
from ..models import Chunk

Content = Literal["embed_text", "text"]
# Metadata the adapters add. Values are flat (str/int/bool): several vector stores reject lists.
MK_KEYS = ("chunk_id", "section", "start_index", "end_index", "token_count", "oversized")


def chunk_for(text: str, doc_id: str, *, max_tokens: int | None, prefix: bool,
              tokenizer: Tokenizer, headerless: str, chunk_kwargs: dict[str, Any]) -> list[Chunk]:
    return chunk(text, doc_id, max_tokens=max_tokens, prefix=prefix, tokenizer=tokenizer,
                 headerless=headerless, **chunk_kwargs)  # type: ignore[arg-type]


def check_content(content: str) -> Content:
    if content not in ("embed_text", "text"):
        raise ValueError("content must be 'embed_text' or 'text'")
    return content  # type: ignore[return-value]


def content_of(c: Chunk, content: Content) -> str:
    return c.embed_text if content == "embed_text" else c.text


def mk_metadata(c: Chunk) -> dict[str, Any]:
    return {"chunk_id": c.chunk_id, "section": " > ".join(c.section_path),
            "start_index": c.start_char, "end_index": c.end_char,
            "token_count": c.token_count, "oversized": c.oversized}
