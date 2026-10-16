"""Shared helpers for the Phase 0 (zero-LLM-cost) re-analyses of the v3 PILOT data.

Everything here reads the existing caches (data/cache, data/emb) read-only.
New embeddings (texts never embedded before, or a new embedder) are cached in
data/emb_phase0/ so the original embedding cache is never written to.
Scoring reuses the v3 harness functions from evaluate.py / evaluate_corpus.py.
"""
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import (CACHE, ENC, covered, render, retrieved_units, unit_grams,  # noqa: E402
                      words, ngrams)
from evaluate import question_metrics as qasper_question_metrics  # noqa: E402
from evaluate_corpus import doc_id, question_metrics as fs_question_metrics  # noqa: E402

OUT = ROOT / "results" / "phase0"
OUT.mkdir(parents=True, exist_ok=True)
EMB_OLD = ROOT / "data" / "emb"            # read-only
EMB_NEW = ROOT / "data" / "emb_phase0"     # new vectors only


def _ollama_url() -> str:
    for u in ("http://127.0.0.1:11435", "http://localhost:11434"):
        try:
            if requests.get(f"{u}/api/version", timeout=3).ok:
                return u
        except requests.RequestException:
            pass
    raise RuntimeError("no Ollama server")


OLLAMA = os.environ.get("OLLAMA_URL") or _ollama_url()

# key -> (ollama model, query prefix, doc prefix, extra options)
EMBEDDERS = {
    "mxbai": ("mxbai-embed-large", "Represent this sentence for searching relevant passages: ", "", {}),
    "nomic": ("nomic-embed-text", "search_query: ", "search_document: ", {}),
    # Qwen3-Embedding: instruction on the query side only (model card convention); 32k context
    "qwen3e": ("qwen3-embedding:0.6b",
               "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", "",
               {"num_ctx": 8192}),
}


def embed(model_key: str, texts: list[str], is_query: bool) -> np.ndarray:
    model, qpre, dpre, opts = EMBEDDERS[model_key]
    pre = qpre if is_query else dpre
    EMB_NEW.mkdir(parents=True, exist_ok=True)
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        h = hashlib.sha1(f"{model}|{pre}{t}".encode()).hexdigest()  # same key as evaluate.embed
        for d in (EMB_OLD, EMB_NEW):
            p = d / f"{h}.npy"
            if p.exists():
                out[i] = np.load(p)
                break
        else:
            todo.append((i, EMB_NEW / f"{h}.npy"))
    for b in range(0, len(todo), 16):
        batch = todo[b:b + 16]
        payload = {"model": model, "input": [pre + texts[i] for i, _ in batch], "truncate": True}
        if opts:
            payload["options"] = opts
        r = requests.post(f"{OLLAMA}/api/embed", json=payload, timeout=900).json()
        for (i, p), v in zip(batch, r["embeddings"]):
            arr = np.asarray(v, dtype=np.float32)
            arr /= np.linalg.norm(arr) + 1e-9
            np.save(p, arr)
            out[i] = arr
    return np.vstack(out)


# ---------------------------------------------------------------- BM25
class BM25:
    """Okapi BM25 identical to rank_bm25.BM25Okapi (epsilon floor on negative IDF),
    but IDF / avgdl can come from an external corpus (corpus-level statistics)."""

    def __init__(self, docs_tok: list[list[str]], k1=1.5, b=0.75, epsilon=0.25,
                 stats_corpus: list[list[str]] | None = None):
        self.k1, self.b = k1, b
        stats = stats_corpus if stats_corpus is not None else docs_tok
        N = len(stats)
        df = Counter(w for d in stats for w in set(d))
        self.avgdl = sum(len(d) for d in stats) / N
        idf = {w: math.log(N - n + 0.5) - math.log(n + 0.5) for w, n in df.items()}
        avg_idf = sum(idf.values()) / len(idf)
        eps = epsilon * avg_idf
        self.idf = {w: (v if v >= 0 else eps) for w, v in idf.items()}
        self.docs = [Counter(d) for d in docs_tok]
        self.lens = [len(d) for d in docs_tok]

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for w in q:
            idf = self.idf.get(w, 0.0)
            if not idf:
                continue
            for j, (tf, dl) in enumerate(zip(self.docs, self.lens)):
                f = tf.get(w, 0)
                if f:
                    s[j] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def tok(d: str) -> list[str]:
    return words(d) or ["_"]


def rrf(*orders_lists):
    fused = []
    for orders in zip(*orders_lists):
        score: dict[int, float] = {}
        for r_ in orders:
            for pos, idx in enumerate(r_):
                score[int(idx)] = score.get(int(idx), 0.0) + 1.0 / (60 + pos + 1)
        fused.append(sorted(score, key=lambda i: -score[i]))
    return fused


def rank(retriever: str, docs: list[str], queries: list[str], bm25_kw: dict | None = None,
         stats_corpus: list[list[str]] | None = None) -> list[list[int]]:
    """retriever in {bm25, mxbai, nomic, qwen3e, hybrid (bm25+nomic), hybrid_qwen3e (bm25+qwen3e)}."""
    if retriever == "bm25":
        if bm25_kw is None and stats_corpus is None:  # exact v3 path
            from rank_bm25 import BM25Okapi
            bm = BM25Okapi([tok(d) for d in docs])
            return [list(np.argsort(-bm.get_scores(words(q)), kind="stable")) for q in queries]
        bm = BM25([tok(d) for d in docs], stats_corpus=stats_corpus, **(bm25_kw or {}))
        return [list(np.argsort(-bm.scores(words(q)), kind="stable")) for q in queries]
    if retriever in EMBEDDERS:
        D = embed(retriever, docs, False)
        Q = embed(retriever, queries, True)
        return [list(np.argsort(-(D @ q), kind="stable")) for q in Q]
    if retriever == "hybrid":
        return rrf(rank("bm25", docs, queries, bm25_kw, stats_corpus), rank("nomic", docs, queries))
    if retriever == "hybrid_qwen3e":
        return rrf(rank("bm25", docs, queries, bm25_kw, stats_corpus), rank("qwen3e", docs, queries))
    raise ValueError(retriever)


# ---------------------------------------------------------------- data
def load_qasper() -> list[dict]:
    """The 30 pilot papers with the v3 question filter (drop ':::' evidence, drop empty)."""
    papers = json.loads((ROOT / "data" / "subset.json").read_text())
    for p in papers:
        for q in p["questions"]:
            q["evidence"] = [e for e in q["evidence"] if ":::" not in e]
        p["questions"] = [q for q in p["questions"] if q["evidence"]]
    return papers


def chunks_of(cs: str, pid: str) -> list[dict]:
    return json.loads((CACHE / cs / f"{pid}.json").read_text())["chunks"]


def load_freshstack():
    data = json.loads((ROOT / "data" / "freshstack_laravel.json").read_text())
    ids = [doc_id(d["path"]) for d in data["docs"]]
    return data, ids


def fs_chunks(cs: str, ids: list[str]) -> list[dict]:
    return [c for i in ids for c in chunks_of(cs, i)]


def fs_gold(q: dict) -> set:
    return set().union(*(ngrams(words(e["text"])) for e in q["evidence"]))


# ---------------------------------------------------------------- evaluation loops
def eval_qasper(papers, get_chunks, get_docs, retrievers, metrics=("rec@512t",),
                evidence_fn=None, rank_fn=None) -> dict:
    """Return per[(retriever, metric)][pid] -> per-question values.

    get_chunks(pid) -> chunks used for SCORING (source text);
    get_docs(pid, chunks) -> rendered strings used for INDEXING.
    evidence_fn(q, order, chunks) -> dict metric->value overrides the default scorer.
    rank_fn(pid, retriever, docs, queries) overrides the default ranker.
    """
    per: dict = {}
    for p in papers:
        chunks = get_chunks(p["id"])
        docs = get_docs(p["id"], chunks)
        qs = [q["question"] for q in p["questions"]]
        for r in retrievers:
            orders = rank_fn(p["id"], r, docs, qs) if rank_fn else rank(r, docs, qs)
            for q, order in zip(p["questions"], orders):
                m = evidence_fn(q, order, chunks) if evidence_fn else qasper_question_metrics(order, chunks, q["evidence"])
                for metric in metrics:
                    per.setdefault((r, metric), {}).setdefault(p["id"], []).append(m[metric])
    return per


def eval_freshstack(data, chunks, docs, retrievers, metrics=("prec@1024t",), **rank_kw) -> dict:
    per: dict = {}
    qs = data["questions"]
    for r in retrievers:
        orders = rank(r, docs, [q["question"] for q in qs], **rank_kw)
        for q, order in zip(qs, orders):
            m = fs_question_metrics(order, chunks, fs_gold(q))
            for metric in metrics:
                per.setdefault((r, metric), {})[q["qid"]] = [m[metric]]
    return per


def mean_of(pp: dict) -> float:
    v = [x for k in pp for x in pp[k]]
    return sum(v) / len(v)


def save_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
