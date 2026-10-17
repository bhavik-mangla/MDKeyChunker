"""Framework adapters and object coercion.

- ``mdkeychunker.integrations.langchain.MDKeyChunkerSplitter``: a LangChain ``TextSplitter``
  (``pip install "mdkeychunker[langchain]"``).
- ``mdkeychunker.integrations.llamaindex.MDKeyChunkerNodeParser``: a LlamaIndex
  ``NodeParser`` (``pip install "mdkeychunker[llamaindex]"``).
- :func:`to_chunks`: LangChain documents, LlamaIndex nodes, Chonkie chunks, dicts or
  strings to ``Chunk`` (no dependency), for ``mk.diagnose`` and ``mk.evidence_recall``.

The adapter modules import their framework; this package does not.
"""
from ._coerce import to_chunks

__all__ = ["to_chunks"]
