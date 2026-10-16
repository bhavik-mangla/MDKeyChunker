"""Dense retrieval over chunks with sentence-transformers (``pip install "mdkeychunker[dense]"``).

Presets carry the query/document prefixes the study used. The study served these
embedders through Ollama; sentence-transformers runs the Hugging Face weights, so scores
are close but not bit-identical to the paper's.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import Chunk
from ._base import BaseRetriever, argsort_desc, index_text, require
from .fusion import Ranking

# Qwen3-Embedding model card: queries get f"Instruct: {task}\nQuery:", documents nothing.
# Default retrieval task from the card, as in the study (retrieval_extra.QWEN3_QUERY_PREFIX).
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_QUERY_PREFIX = f"Instruct: {QWEN3_TASK}\nQuery:"


@dataclass(frozen=True)
class DensePreset:
    """A Hugging Face model id with the prefixes it expects."""

    model_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False
    max_seq_length: int | None = None


DENSE_PRESETS: dict[str, DensePreset] = {
    # strongest dense retriever in the study (secondary/exploratory findings 5, 7)
    "qwen3-embedding-0.6b": DensePreset("Qwen/Qwen3-Embedding-0.6B", QWEN3_QUERY_PREFIX, "",
                                        max_seq_length=8192),
    # the dense half of the study's primary hybrid
    "nomic": DensePreset("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: ",
                         trust_remote_code=True, max_seq_length=8192),
    "mxbai": DensePreset("mixedbread-ai/mxbai-embed-large-v1",
                         "Represent this sentence for searching relevant passages: ", ""),
}

EmbedFn = Callable[[list[str]], Any]  # texts -> 2-D array-like of embeddings


def resolve_dense(model: str) -> DensePreset:
    """A preset by name, or a bare Hugging Face model id with no prefixes."""
    return DENSE_PRESETS.get(model) or DensePreset(model)


class DenseRetriever(BaseRetriever):
    """Cosine similarity between query and chunk embeddings (``embed_text`` or ``text``).

    Args:
        chunks: the chunks to index (embedded once, at construction).
        model: a preset name (``"qwen3-embedding-0.6b"``, ``"nomic"``, ``"mxbai"``) or a
            Hugging Face model id.
        embed_fn: bring your own embedder: a function from a list of (prefixed) texts to
            a 2-D array of embeddings. ``model`` is then only a label. Needs numpy.
        query_prefix, doc_prefix: override the preset's prefixes.
        device, batch_size, max_seq_length: passed to sentence-transformers.

    Example (no model download; a toy embedder):
        >>> from mdkeychunker.models import Chunk
        >>> fake = lambda texts: [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]
        >>> r = DenseRetriever([Chunk(text="a dog"), Chunk(text="a cat")], embed_fn=fake)
        >>> r.search("cat?", k=1)[0].chunk.text
        'a cat'
    """

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model: str = "qwen3-embedding-0.6b", *,
                 embed_fn: EmbedFn | None = None, query_prefix: str | None = None,
                 doc_prefix: str | None = None, device: str | None = None,
                 batch_size: int = 32, max_seq_length: int | None = None) -> None:
        super().__init__(chunks)
        self._np = require("numpy", "dense")
        preset = resolve_dense(model)
        self.model = model
        self.query_prefix = preset.query_prefix if query_prefix is None else query_prefix
        self.doc_prefix = preset.doc_prefix if doc_prefix is None else doc_prefix
        self.batch_size = batch_size
        if embed_fn is None:
            st = require("sentence_transformers", "dense")
            enc = st.SentenceTransformer(preset.model_id, device=device,
                                         trust_remote_code=preset.trust_remote_code)
            seq = max_seq_length or preset.max_seq_length
            if seq:
                enc.max_seq_length = seq
            self._model: Any = enc

            def st_embed(texts: list[str]) -> Any:
                return enc.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
            embed_fn = st_embed
        self._embed = embed_fn
        self.doc_embeddings = self._encode([self.doc_prefix + index_text(c) for c in self.chunks])

    def _encode(self, texts: list[str]) -> Any:
        np = self._np
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(self._embed(texts), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValueError(f"embed_fn returned shape {arr.shape} for {len(texts)} texts")
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    def rank_many(self, queries: Sequence[str]) -> list[Ranking]:
        if not self.chunks:
            return [[] for _ in queries]
        qs = self._encode([self.query_prefix + q for q in queries])
        return [argsort_desc(self.doc_embeddings @ q) for q in qs]
