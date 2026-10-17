"""Rank fusion: reciprocal rank fusion (RRF) and convex combination of normalised scores.

Pure Python, no dependencies. A ranking is a list of ``(index, score)`` pairs, best
first; fusion returns one in the same form.

``RRF(k=60)`` with equal weights is the study's hybrid (``benchmarks/chunking_study/
evaluate_v2.rrf``), including its tie order: items with equal fused score keep the
order in which they first appear, scanning the first input list, then the second.

Example:
    >>> RRF().fuse([[(0, 9.0), (1, 5.0)], [(1, 0.8), (2, 0.7)]])[0][0]
    1
    >>> [i for i, _ in Convex(alpha=0.7).fuse([[(0, 9.0), (1, 5.0)], [(1, 0.8), (2, 0.7)]])]
    [1, 0, 2]
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Ranking = list[tuple[int, float]]

__all__ = ["RRF", "Convex", "Fusion", "Ranking"]


class Fusion(Protocol):
    """Anything that fuses several rankings into one."""

    def fuse(self, rankings: Sequence[Sequence[tuple[int, float]]]) -> Ranking: ...


def _weights(weights: Sequence[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(f"got {len(weights)} weights for {n} rankings")
    if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
        raise ValueError("weights must be >= 0 and not all zero")
    return [float(w) for w in weights]


def _ordered(score: dict[int, float]) -> Ranking:
    # sorted() is stable, so ties keep first-appearance order (the study's behaviour)
    return sorted(score.items(), key=lambda kv: -kv[1])


@dataclass(frozen=True)
class RRF:
    """Reciprocal rank fusion: ``sum_i w_i / (k + rank_i)``, rank starting at 1.

    Input scores are ignored; only ranks matter. An item missing from a list gets nothing
    from it. Untuned ``k=60`` with equal weights is what the study's hybrid used; on Qasper
    that hybrid scored below its own dense half (42.7 vs 47.8 recall@512, free prefix;
    exploratory finding), so check yours with ``mk.diagnose`` before relying on it.

    Example:
        >>> RRF(k=0).fuse([[(7, 1.0), (8, 0.5)]])
        [(7, 1.0), (8, 0.5)]
    """

    k: float = 60.0
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be >= 0")

    def fuse(self, rankings: Sequence[Sequence[tuple[int, float]]]) -> Ranking:
        ws = _weights(self.weights, len(rankings))
        score: dict[int, float] = {}
        for w, ranking in zip(ws, rankings):
            for pos, (idx, _) in enumerate(ranking):
                score[idx] = score.get(idx, 0.0) + w / (self.k + pos + 1)
        return _ordered(score)


Norm = Literal["minmax", "zscore"]


def _normalise(values: list[float], norm: Norm) -> list[float]:
    if not values:
        return []
    if norm == "minmax":
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) if hi > lo else 1.0 for v in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / sd if sd > 0 else 0.0 for v in values]


@dataclass(frozen=True)
class Convex:
    """Convex combination of per-list normalised scores.

    With two lists (sparse first, dense second, as ``HybridRetriever`` passes them), the
    fused score is ``(1 - alpha) * sparse + alpha * dense``, so ``alpha=1`` is dense only.
    For more lists give ``weights`` instead. Scores are normalised within each list
    (``"minmax"`` to [0, 1], or ``"zscore"``). A list whose scores are all equal
    normalises to 1.0 (minmax) or 0.0 (zscore) for every item. An item missing from a
    list scores 0.0 there (minmax), or the list's lowest z-score (-1.0 if all are equal).

    Example:
        >>> Convex(alpha=1.0).fuse([[(0, 3.0), (1, 1.0)], [(1, 0.9), (0, 0.1)]])[0][0]
        1
    """

    alpha: float = 0.5
    norm: Norm = "minmax"
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.norm not in ("minmax", "zscore"):
            raise ValueError("norm must be 'minmax' or 'zscore'")

    def fuse(self, rankings: Sequence[Sequence[tuple[int, float]]]) -> Ranking:
        n = len(rankings)
        if self.weights is not None:
            ws = _weights(self.weights, n)
        elif n == 2:
            ws = [1.0 - self.alpha, self.alpha]
        elif n == 1:
            ws = [1.0]
        else:
            raise ValueError("Convex with more than two rankings needs weights")
        order: dict[int, None] = {}
        normed: list[dict[int, float]] = []
        for ranking in rankings:
            vals = _normalise([s for _, s in ranking], self.norm)
            normed.append({idx: v for (idx, _), v in zip(ranking, vals)})
            for idx, _ in ranking:
                order.setdefault(idx, None)
        floors = [self._floor(m) for m in normed]
        score = {idx: sum(w * m.get(idx, f) for w, m, f in zip(ws, normed, floors))
                 for idx in order}
        return _ordered(score)

    def _floor(self, normed: dict[int, float]) -> float:
        """Score for an item missing from a list: never above the list's worst item."""
        if self.norm == "minmax" or not normed:
            return 0.0
        lo = min(normed.values())
        return lo if lo < max(normed.values()) else -1.0
