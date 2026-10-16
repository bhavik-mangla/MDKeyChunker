"""Shared retriever plumbing: what text is indexed, rankings -> ``Hit`` lists, lazy imports."""
from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from ..models import Chunk, Hit
from .fusion import Ranking


def index_text(chunk: Chunk) -> str:
    """The text a retriever indexes: ``embed_text`` (prefix + text) when set, else ``text``."""
    return chunk.embed_text or chunk.text


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency or fail with the ``pip install`` line that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f'{module} is needed for this retriever: pip install "mdkeychunker[{extra}]"') from e


class BaseRetriever:
    """A retriever over a fixed list of chunks.

    Subclasses implement ``rank_many(queries)``, returning for each query a ranking of
    ``(chunk index, score)`` pairs, best first. ``search`` turns rankings into ``Hit``
    objects whose ``ranks`` holds this retriever's 1-based rank under ``name``.
    """

    name: str = "retriever"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: list[Chunk] = list(chunks)
        for c in self.chunks:
            if not isinstance(c, Chunk):
                raise TypeError(f"expected Chunk objects, got {type(c).__name__}")

    def __len__(self) -> int:
        return len(self.chunks)

    def rank_many(self, queries: Sequence[str]) -> list[Ranking]:
        raise NotImplementedError

    def rank(self, query: str) -> Ranking:
        return self.rank_many([query])[0]

    def _hits(self, ranking: Ranking, k: int, extra: dict[int, dict[str, int]] | None = None
              ) -> list[Hit]:
        out = []
        for pos, (idx, score) in enumerate(ranking[:k]):
            ranks = {self.name: pos + 1}
            if extra and idx in extra:
                ranks = {**extra[idx], **ranks}
            out.append(Hit(chunk=self.chunks[idx], score=float(score), ranks=ranks))
        return out

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top ``k`` hits for one query."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[Hit]]:
        """Top ``k`` hits for each query (batched where the backend allows)."""
        if k < 0:
            raise ValueError("k must be >= 0")
        return [self._hits(r, k) for r in self.rank_many(list(queries))]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, chunks={len(self.chunks)})"


def argsort_desc(scores: Any) -> Ranking:
    """Indices by descending score; ties keep index order (numpy stable argsort, as the study)."""
    import numpy as np

    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    return [(int(i), float(arr[i])) for i in order]
