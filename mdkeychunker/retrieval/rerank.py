"""Cross-encoder reranking (``pip install "mdkeychunker[rerank]"``).

Ported from the study (``benchmarks/chunking_study/retrieval_extra.py``): the Qwen3
reranker's chat template applied by hand, length-sorted batches, ties kept in first-stage
order, and the same optional on-disk score cache layout and key, so a cache written by
the study harness is read as is.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._base import require

QWEN3_RERANKER = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_PRESETS = {"qwen3-reranker-0.6b": QWEN3_RERANKER, "bge-reranker-v2-m3": BGE_RERANKER}
RERANK_DEPTH = 30  # the study reranked the hybrid top-30

RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
              'the Query and the Instruct provided. Note that the answer can only be "yes" or '
              '"no".<|im_end|>\n<|im_start|>user\n')
_QR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_VERSION = "v1"

ScoreFn = Callable[[list[tuple[str, str]]], Sequence[float]]


def format_pair(model_name: str, query: str, doc: str) -> tuple[str, str]:
    """Apply the Qwen3 reranker's chat template by hand (model card); others take raw text.

    Example:
        >>> format_pair("BAAI/bge-reranker-v2-m3", "q", "d")
        ('q', 'd')
    """
    if "qwen3-reranker" in model_name.lower():
        return (f"{_QR_PREFIX}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n",
                f"<Document>: {doc}{_QR_SUFFIX}")
    return query, doc


def rerank_order(order: Sequence[int], head_scores: Sequence[float], top: int) -> list[int]:
    """Reorder ``order[:top]`` by score (descending; ties keep the first-stage rank).

    Example:
        >>> rerank_order([5, 6, 7, 8], [0.1, 0.9, 0.1], top=3)
        [6, 5, 7, 8]
    """
    head = list(order[:top])
    if len(head_scores) != len(head):
        raise ValueError("need one score per reranked item")
    idx = sorted(range(len(head)), key=lambda j: (-head_scores[j], j))
    return [head[j] for j in idx] + list(order[top:])


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class RerankCache:
    """Scores on disk: ``<root>/<sha1(model_key)[:16]>/<qh[:2]>/<qh>.json = {text_hash: score}``."""

    def __init__(self, root: str | Path, model_key: str) -> None:
        self.root = Path(root) / _sha1(model_key)[:16]
        self.model_key = model_key
        self._mem: dict[str, dict[str, float]] = {}

    @staticmethod
    def query_hash(query: str) -> str:
        return _sha1("q|" + query)

    @staticmethod
    def text_hash(text: str) -> str:
        return _sha1("t|" + text)

    def _path(self, qh: str) -> Path:
        return self.root / qh[:2] / f"{qh}.json"

    def load(self, query: str) -> dict[str, float]:
        qh = self.query_hash(query)
        if qh not in self._mem:
            p = self._path(qh)
            self._mem[qh] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return self._mem[qh]

    def save(self, query: str, scores: dict[str, float]) -> None:
        cur = self.load(query)
        cur.update(scores)
        p = self._path(self.query_hash(query))
        p.parent.mkdir(parents=True, exist_ok=True)
        key_file = self.root / "model_key.txt"
        if not key_file.exists():
            key_file.write_text(self.model_key, encoding="utf-8")
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(cur), encoding="utf-8")
        tmp.replace(p)


class Reranker:
    """Cross-encoder reranker (sentence-transformers ``CrossEncoder``), loaded lazily.

    Args:
        model: a preset (``"qwen3-reranker-0.6b"``, ``"bge-reranker-v2-m3"``) or a model id.
        max_length, batch_size, device: passed to ``CrossEncoder``.
        cache_dir: optional on-disk score cache (study layout); ``None`` keeps scores in memory.
        score_fn: bring your own scorer: formatted ``(query, doc)`` pairs -> scores. No
            model is loaded.

    Example (no model download):
        >>> r = Reranker(score_fn=lambda pairs: [float("cat" in d) for _, d in pairs])
        >>> r.scores("which pet?", ["a dog", "a cat"])
        [0.0, 1.0]
    """

    def __init__(self, model: str = "qwen3-reranker-0.6b", *, max_length: int = 2048,
                 batch_size: int = 16, device: str | None = None,
                 cache_dir: str | Path | None = None, score_fn: ScoreFn | None = None) -> None:
        self.model_name = RERANK_PRESETS.get(model, model)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.score_fn = score_fn
        self._ce: Any = None
        self._cache = RerankCache(cache_dir, self.model_key) if cache_dir is not None else None
        self._mem: dict[tuple[str, str], float] = {}
        self.pairs_scored = 0

    @property
    def model_key(self) -> str:
        return f"{self.model_name}|tmpl={TEMPLATE_VERSION}|maxlen={self.max_length}"

    def _load(self) -> Any:
        st = require("sentence_transformers", "rerank")
        ce = st.CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        tok = ce.tokenizer
        if tok.pad_token is None:  # Qwen3 has no pad token; batching needs one
            tok.pad_token = tok.eos_token
        return ce

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.score_fn is not None:
            return [float(s) for s in self.score_fn(pairs)]
        if self._ce is None:
            self._ce = self._load()
        # longest first so batches pad little; results go back to input order
        perm = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        out = self._ce.predict([pairs[i] for i in perm], batch_size=self.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
        res = [0.0] * len(pairs)
        for j, i in enumerate(perm):
            res[i] = float(out[j].reshape(-1)[0]) if hasattr(out[j], "reshape") else float(out[j])
        return res

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance score of each text for ``query`` (higher is more relevant)."""
        if self._cache is not None:
            cached = self._cache.load(query)
            hs = [RerankCache.text_hash(t) for t in texts]
            todo = sorted({h: t for h, t in zip(hs, texts) if h not in cached}.items())
            if todo:
                vals = self._predict([format_pair(self.model_name, query, t) for _, t in todo])
                self.pairs_scored += len(todo)
                self._cache.save(query, {h: v for (h, _), v in zip(todo, vals)})
                cached = self._cache.load(query)
            return [cached[h] for h in hs]
        new = list(dict.fromkeys(t for t in texts if (query, t) not in self._mem))
        if new:
            vals = self._predict([format_pair(self.model_name, query, t) for t in new])
            self.pairs_scored += len(new)
            self._mem.update({(query, t): v for t, v in zip(new, vals)})
        return [self._mem[(query, t)] for t in texts]
