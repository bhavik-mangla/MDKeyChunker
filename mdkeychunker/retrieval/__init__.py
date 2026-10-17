"""Reference retrievers for evaluation: BM25 (bm25s), dense (sentence-transformers), fusion, rerank.

Importing this package needs no optional dependency; each retriever imports its backend
when constructed and names the extra to install if it is missing:
``pip install "mdkeychunker[bm25]"``, ``[dense]``, ``[rerank]`` or ``[retrieval]`` (all three).
"""
from .bm25 import BM25Retriever
from .dense import DENSE_PRESETS, DensePreset, DenseRetriever, load_embedder
from .fusion import RRF, Convex, Fusion
from .hybrid import HybridRetriever
from .rerank import RERANK_PRESETS, Reranker

__all__ = [
    "BM25Retriever",
    "DENSE_PRESETS",
    "DensePreset",
    "DenseRetriever",
    "Convex",
    "Fusion",
    "HybridRetriever",
    "RERANK_PRESETS",
    "RRF",
    "Reranker",
    "load_embedder",
]
