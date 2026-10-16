"""Hybrid retrieval: BM25 and/or dense, fused, optionally reranked."""
from __future__ import annotations

from collections.abc import Sequence

from ..models import Chunk, Hit
from ._base import BaseRetriever, index_text
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import RRF, Fusion, Ranking
from .rerank import RERANK_DEPTH, Reranker, rerank_order


class HybridRetriever(BaseRetriever):
    """Fuse BM25 and dense rankings, then optionally rerank the top of the fused list.

    A reference retriever for evaluation, not a production engine: for production, use your
    vector database's hybrid search and use ``mk.diagnose`` to choose its settings.

    Args:
        chunks: the chunks to index.
        bm25: ``True`` (build a ``BM25Retriever``), ``False``, or a prebuilt one.
        dense: a preset name / model id, a prebuilt ``DenseRetriever``, or ``None``.
            Prebuilt retrievers must index the same chunks; pass them to reuse an index
            across several fusion settings.
        fusion: ``RRF()`` (default; the study's untuned k=60) or ``Convex(...)``. Lists are
            passed sparse first, dense second.
        rerank: a reranker preset / model id, a ``Reranker``, or ``None``.
        rerank_depth: how many fused results the reranker reorders (study: 30).
        depth: candidates taken from each list before fusion (``None`` = all chunks, as in
            the study's within-paper search).

    ``Hit.ranks`` holds the 1-based rank in each input list (``"bm25"``, ``"dense"``), the
    fused rank (``"fused"``) and, for reranked hits, ``"rerank"``. ``Hit.score`` is the
    reranker score for reranked hits and the fused score otherwise.

    Example (no model download; a toy embedder):
        >>> from mdkeychunker.models import Chunk
        >>> from mdkeychunker.retrieval.dense import DenseRetriever
        >>> cs = [Chunk(text="install on linux"), Chunk(text="pricing plans")]
        >>> d = DenseRetriever(cs, embed_fn=lambda ts: [[float("linux" in t), 1.0] for t in ts])
        >>> h = HybridRetriever(cs, dense=d).search("linux setup", k=1)[0]
        >>> h.chunk.text, sorted(h.ranks)
        ('install on linux', ['bm25', 'dense', 'fused'])
    """

    def __init__(self, chunks: Sequence[Chunk], *, bm25: bool | BM25Retriever = True,
                 dense: str | DenseRetriever | None = "qwen3-embedding-0.6b",
                 fusion: Fusion | None = None, rerank: str | Reranker | None = None,
                 rerank_depth: int = RERANK_DEPTH, depth: int | None = None) -> None:
        super().__init__(chunks)
        self.parts: list[BaseRetriever] = []
        if isinstance(bm25, BM25Retriever):
            self.parts.append(self._check(bm25))
        elif bm25:
            self.parts.append(BM25Retriever(self.chunks))
        if isinstance(dense, DenseRetriever):
            self.parts.append(self._check(dense))
        elif dense:
            self.parts.append(DenseRetriever(self.chunks, dense))
        if not self.parts:
            raise ValueError("HybridRetriever needs bm25 and/or dense")
        self.fusion: Fusion = fusion or RRF()
        self.reranker = Reranker(rerank) if isinstance(rerank, str) else rerank
        if rerank_depth < 1:
            raise ValueError("rerank_depth must be >= 1")
        self.rerank_depth = rerank_depth
        self.depth = depth
        self.name = self._name()

    def _check(self, r: BaseRetriever) -> BaseRetriever:
        same = len(r.chunks) == len(self.chunks) and all(
            a is b for a, b in zip(r.chunks, self.chunks))
        if not same:
            raise ValueError(f"prebuilt {r.name} retriever indexes different chunks")
        return r

    def _name(self) -> str:
        base = "+".join(p.name for p in self.parts)
        return f"{base}+rerank" if self.reranker else base

    def _ranked(self, queries: Sequence[str]) -> list[tuple[Ranking, dict[int, dict[str, int]]]]:
        per_part = [p.rank_many(queries) for p in self.parts]
        out = []
        for qi in range(len(queries)):
            lists = [r[qi][: self.depth] if self.depth else r[qi] for r in per_part]
            ranks: dict[int, dict[str, int]] = {}
            for part, lst in zip(self.parts, lists):
                for pos, (idx, _) in enumerate(lst):
                    ranks.setdefault(idx, {})[part.name] = pos + 1
            fused = self.fusion.fuse(lists) if len(lists) > 1 else list(lists[0])
            for pos, (idx, _) in enumerate(fused):
                ranks[idx]["fused"] = pos + 1
            out.append((fused, ranks))
        return out

    def rank_many(self, queries: Sequence[str]) -> list[Ranking]:
        return [self._rerank(q, fused, ranks) for q, (fused, ranks)
                in zip(queries, self._ranked(queries))]

    def _rerank(self, query: str, fused: Ranking, ranks: dict[int, dict[str, int]]) -> Ranking:
        if not self.reranker or not fused:
            return fused
        head = fused[: self.rerank_depth]
        scores = self.reranker.scores(query, [index_text(self.chunks[i]) for i, _ in head])
        by_idx = {i: s for (i, _), s in zip(head, scores)}
        order = rerank_order([i for i, _ in fused], scores, self.rerank_depth)
        tail = dict(fused[self.rerank_depth:])
        out = []
        for pos, idx in enumerate(order):
            if idx in by_idx:
                ranks[idx]["rerank"] = pos + 1
                out.append((idx, by_idx[idx]))
            else:
                out.append((idx, tail[idx]))
        return out

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[Hit]]:
        if k < 0:
            raise ValueError("k must be >= 0")
        out = []
        for q, (fused, ranks) in zip(queries, self._ranked(list(queries))):
            ranking = self._rerank(q, fused, ranks)
            out.append([Hit(chunk=self.chunks[i], score=float(s), ranks=dict(ranks[i]))
                        for i, s in ranking[:k]])
        return out
