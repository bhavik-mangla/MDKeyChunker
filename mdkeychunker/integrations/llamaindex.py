"""LlamaIndex ``NodeParser`` backed by ``mk.chunk`` (``pip install "mdkeychunker[llamaindex]"``).

>>> from mdkeychunker.integrations.llamaindex import MDKeyChunkerNodeParser  # doctest: +SKIP
>>> nodes = MDKeyChunkerNodeParser(max_tokens=512).get_nodes_from_documents(docs)  # doctest: +SKIP
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, TextNode
    from llama_index.core.utils import get_tqdm_iterable
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('The LlamaIndex adapter needs: pip install "mdkeychunker[llamaindex]"') from e

from ._common import MK_KEYS, check_content, chunk_for, content_of, mk_metadata

__all__ = ["MDKeyChunkerNodeParser"]


class MDKeyChunkerNodeParser(NodeParser):
    """Header-first chunks capped at ``max_tokens``, as LlamaIndex ``TextNode`` objects.

    With the default ``content="embed_text"`` the node text is ``Chunk.embed_text`` (section
    path, blank line, text: the v4 study's free prefix) and the adapter's own metadata keys
    are excluded from the embed and LLM views, so the embedded string is exactly what
    ``mk.chunk`` would embed (plus any metadata of your documents, as LlamaIndex does).
    With ``content="text"`` the node holds the raw text and LlamaIndex embeds the path from
    ``metadata["section"]`` in its own ``key: value`` format.

    Node ids are the content-stable ``chunk_id``; ``start_char_idx`` / ``end_char_idx`` are
    offsets into the source document. Prev/next and source relationships are set by
    LlamaIndex as for its own parsers.
    """

    max_tokens: int | None = Field(default=512, description="cap on each chunk's embed_text")
    prefix: bool = Field(default=True, description="section-path prefix in embed_text")
    content: str = Field(default="embed_text", description="'embed_text' or 'text'")
    headerless: str = Field(default="auto", description="see mk.chunk")
    tokenizer: Any = Field(default=None, exclude=True, description="see mk.chunk")
    chunk_kwargs: dict[str, Any] = Field(default_factory=dict, description="more mk.chunk args")

    @classmethod
    def class_name(cls) -> str:
        return "MDKeyChunkerNodeParser"

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False,
                     **kwargs: Any) -> list[BaseNode]:
        content = check_content(self.content)
        hidden = list(MK_KEYS) if content == "embed_text" else \
            [k for k in MK_KEYS if k != "section"]
        out: list[BaseNode] = []
        for node in get_tqdm_iterable(nodes, show_progress, "Chunking with mdkeychunker"):
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            for c in chunk_for(text, node.node_id, max_tokens=self.max_tokens,
                               prefix=self.prefix, tokenizer=self.tokenizer,
                               headerless=self.headerless, chunk_kwargs=self.chunk_kwargs):
                out.append(TextNode(
                    id_=c.chunk_id, text=content_of(c, content), metadata=mk_metadata(c),
                    excluded_embed_metadata_keys=list(hidden),
                    excluded_llm_metadata_keys=list(hidden),
                    start_char_idx=c.start_char, end_char_idx=c.end_char,
                    relationships={NodeRelationship.SOURCE: node.as_related_node_info()}))
        return out
