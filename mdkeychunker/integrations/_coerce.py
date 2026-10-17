"""Turn other libraries' chunk objects into ``Chunk`` (duck-typed; imports nothing optional)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..models import Chunk

_TEXT_KEYS = ("text", "page_content", "content")
_ID_KEYS = ("chunk_id", "id", "id_", "node_id")
_DOC_KEYS = ("doc_id", "ref_doc_id", "source", "document_id")


def _first(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        v = mapping.get(k)
        if v not in (None, ""):
            return v
    return default


def _from_dict(d: Mapping[str, Any]) -> Chunk:
    meta = dict(d.get("metadata") or {})
    text = _first(d, _TEXT_KEYS)
    if not isinstance(text, str):
        raise TypeError(f"chunk dict needs one of {_TEXT_KEYS} as a string: {sorted(d)}")
    start = _first(d, ("start_char", "start_index", "start_char_idx"),
                   meta.get("start_index", 0))
    end = _first(d, ("end_char", "end_index", "end_char_idx"))
    path = d.get("section_path", [])
    c = Chunk(text=text, embed_text=str(d.get("embed_text") or text),
              doc_id=str(_first(d, _DOC_KEYS, _first(meta, _DOC_KEYS, ""))),
              start_char=int(start),
              end_char=int(end) if end is not None else int(start) + len(text),
              section_path=[str(p) for p in path] if isinstance(path, (list, tuple)) else [],
              section_title=str(d.get("section_title", "")), metadata=meta)
    c.chunk_id = str(_first(d, _ID_KEYS, "")) or c.stable_id()
    return c


def _from_langchain(doc: Any) -> Chunk:
    meta = dict(doc.metadata or {})
    return _from_dict({"text": doc.page_content, "id": getattr(doc, "id", None),
                       "start_index": meta.get("start_index"), "metadata": meta,
                       "chunk_id": meta.get("chunk_id")})


def _from_llamaindex(node: Any) -> Chunk:
    from llama_index.core.schema import MetadataMode  # only reached for LlamaIndex nodes

    meta = dict(node.metadata or {})
    text = node.get_content(metadata_mode=MetadataMode.NONE)
    c = _from_dict({"text": text, "embed_text": node.get_content(metadata_mode=MetadataMode.EMBED),
                    "node_id": node.node_id, "ref_doc_id": node.ref_doc_id,
                    "start_char_idx": getattr(node, "start_char_idx", None),
                    "end_char_idx": getattr(node, "end_char_idx", None), "metadata": meta})
    return c


def _from_attrs(obj: Any) -> Chunk:
    """Chonkie-style chunks and anything else with a ``text`` attribute."""
    return _from_dict({k: getattr(obj, k) for k in
                       ("text", "id", "start_index", "end_index", "doc_id", "embed_text",
                        "section_path", "metadata") if getattr(obj, k, None) is not None})


def to_chunks(items: Iterable[Any]) -> list[Chunk]:
    """Coerce chunks from other libraries for ``mk.diagnose`` / ``mk.evidence_recall``.

    Accepts ``Chunk``, plain strings, dicts (``text`` / ``page_content`` / ``content``, and
    optionally ids, ``doc_id``, offsets, ``metadata``), LangChain ``Document``, LlamaIndex
    nodes (``embed_text`` = the node's EMBED-mode content) and Chonkie chunks (``text``,
    ``start_index``, ``end_index``). Ids are kept when given, else ``Chunk.stable_id()``.
    Offsets are only as good as the source's: span-based gold needs them, text-based gold
    does not.

    >>> [c.text for c in to_chunks(["a", {"page_content": "b", "metadata": {"source": "d"}}])]
    ['a', 'b']
    >>> to_chunks([{"content": "b", "metadata": {"source": "d"}}])[0].doc_id
    'd'
    """
    out: list[Chunk] = []
    for item in items:
        if isinstance(item, Chunk):
            out.append(item)
        elif isinstance(item, str):
            out.append(_from_dict({"text": item}))
        elif isinstance(item, Mapping):
            out.append(_from_dict(item))
        elif hasattr(item, "page_content"):
            out.append(_from_langchain(item))
        elif hasattr(item, "get_content") and hasattr(item, "node_id"):
            out.append(_from_llamaindex(item))
        elif isinstance(getattr(item, "text", None), str):
            out.append(_from_attrs(item))
        else:
            raise TypeError(f"cannot turn {type(item).__name__} into a Chunk; pass Chunk, str, "
                            "dict, LangChain Document, LlamaIndex node or a chunk with .text")
    return out
