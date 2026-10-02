"""Retrieval evaluation over cached chunk sets.

Retrieval is per paper (Qasper questions are about one paper). Relevance is
chunk-agnostic: an evidence paragraph counts as retrieved when at least half of
its word 5-grams appear in the retrieved text. Budgeted metrics fill a fixed
token budget with retrieved chunk text in rank order, so methods that make
bigger chunks get no free credit.
"""
import argparse
import os
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import requests
import tiktoken
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache"
EMB_CACHE = ROOT / "data" / "emb"
ENC = tiktoken.get_encoding("cl100k_base")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

EMBEDDERS = {
    "mxbai": ("mxbai-embed-large", "Represent this sentence for searching relevant passages: ", ""),
    "nomic": ("nomic-embed-text", "search_query: ", "search_document: "),
}

# (chunk set, how chunks are rendered for indexing)
SYSTEMS = [
    ("fixed512", "text"),
    ("fixedtok", "text"),
    ("struct", "text"),
    ("struct", "tc"),
    ("cr", "cr"),
    ("enr_rk", "meta"),
    ("enr_nork", "meta"),
    ("merged_rk", "text"),
    ("merged_rk", "meta"),
    ("merged_nork", "meta"),
]
BASELINE = ("struct", "text")
TOKEN = re.compile(r"[a-z0-9]+")


def render(c: dict, mode: str) -> str:
    if mode == "tc":
        return f"{c.get('section', '')}\n\n{c['text']}"
    if mode == "cr":
        return f"{c.get('context', '')}\n\n{c['text']}"
    if mode == "meta":
        head = [c.get("title", ""), c.get("summary", ""), ", ".join(c.get("keywords", [])),
                " ".join(c.get("questions", []))]
        return "\n".join(h for h in head if h) + "\n\n" + c["text"]
    return c["text"]


def words(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def ngrams(ws: list[str], n: int = 5) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)} if len(ws) >= n else {tuple(ws)}


def covered(evidence: str, retrieved_ngrams: set, retrieved_words: set) -> bool:
    ws = words(evidence)
    if len(ws) < 5:
        return bool(ws) and all(w in retrieved_words for w in ws)
    grams = ngrams(ws)
    return sum(g in retrieved_ngrams for g in grams) / len(grams) >= 0.5


def embed(model_key: str, texts: list[str], is_query: bool) -> np.ndarray:
    model, qpre, dpre = EMBEDDERS[model_key]
    pre = qpre if is_query else dpre
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        h = hashlib.sha1(f"{model}|{pre}{t}".encode()).hexdigest()
        p = EMB_CACHE / f"{h}.npy"
        if p.exists():
            out[i] = np.load(p)
        else:
            todo.append((i, p))
    for b in range(0, len(todo), 32):
        batch = todo[b:b + 32]
        r = requests.post(f"{OLLAMA}/api/embed", json={
            "model": model, "input": [pre + texts[i] for i, _ in batch], "truncate": True,
        }, timeout=600).json()
        for (i, p), v in zip(batch, r["embeddings"]):
            arr = np.asarray(v, dtype=np.float32)
            arr /= np.linalg.norm(arr) + 1e-9
            np.save(p, arr)
            out[i] = arr
    return np.vstack(out)


def rank(retriever: str, docs: list[str], queries: list[str]) -> list[list[int]]:
    if retriever == "bm25":
        bm = BM25Okapi([words(d) or ["_"] for d in docs])
        return [list(np.argsort(-bm.get_scores(words(q)), kind="stable")) for q in queries]
    if retriever in EMBEDDERS:
        D = embed(retriever, docs, False)
        Q = embed(retriever, queries, True)
        return [list(np.argsort(-(D @ q), kind="stable")) for q in Q]
    if retriever == "hybrid":  # reciprocal rank fusion of BM25 and the long-context embedder
        a, b = rank("bm25", docs, queries), rank("nomic", docs, queries)
        fused = []
        for ra, rb in zip(a, b):
            score: dict[int, float] = {}
            for r_ in (ra, rb):
                for pos, idx in enumerate(r_):
                    score[idx] = score.get(idx, 0.0) + 1.0 / (60 + pos + 1)
            fused.append(sorted(score, key=lambda i: -score[i]))
        return fused
    raise ValueError(retriever)


def question_metrics(order: list[int], chunks: list[dict], evidence: list[str]) -> dict:
    m = {}
    for k in (1, 3, 5):
        text = "\n".join(chunks[i]["text"] for i in order[:k])
        ws = words(text)
        g, wset = ngrams(ws), set(ws)
        cov = [covered(e, g, wset) for e in evidence]
        m[f"hit@{k}"] = float(any(cov))
        m[f"rec@{k}"] = sum(cov) / len(cov)
    for budget in (256, 512, 1024):
        toks: list[int] = []
        for i in order:
            toks += ENC.encode(chunks[i]["text"] + "\n")
            if len(toks) >= budget:
                break
        ws = words(ENC.decode(toks[:budget]))
        g, wset = ngrams(ws), set(ws)
        cov = [covered(e, g, wset) for e in evidence]
        m[f"rec@{budget}t"] = sum(cov) / len(cov)
    return m


def cluster_bootstrap(per_paper: dict[str, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over questions with a 95% CI from resampling papers (questions within a paper correlate)."""
    pids = list(per_paper)
    rng = random.Random(seed)
    allv = [v for p in pids for v in per_paper[p]]
    means = []
    for _ in range(n):
        vals = [v for p in (rng.choice(pids) for _ in pids) for v in per_paper[p]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return sum(allv) / len(allv), means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(ROOT / "data" / "subset.json"))
    ap.add_argument("--retrievers", default="bm25,mxbai,nomic,hybrid")
    ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    args = ap.parse_args()

    papers = json.loads(Path(args.subset).read_text())
    systems = SYSTEMS
    # Compare systems on identical questions: keep papers complete for every system
    complete = [p for p in papers if all((CACHE / s[0] / f"{p['id']}.json").exists() for s in systems)]
    if len(complete) < len(papers):
        print(f"evaluating {len(complete)}/{len(papers)} papers complete for all systems; "
              f"excluded: {sorted(set(p['id'] for p in papers) - set(p['id'] for p in complete))}")
    papers = complete
    print(f"{len(papers)} papers, {sum(len(p['questions']) for p in papers)} questions")

    # per[(retriever, system, metric)][paper_id] -> list of per-question values
    per: dict = {}
    for p in papers:
        qs = [q["question"] for q in p["questions"]]
        for cs, mode in systems:
            chunks = json.loads((CACHE / cs / f"{p['id']}.json").read_text())["chunks"]
            docs = [render(c, mode) for c in chunks]
            for r in args.retrievers.split(","):
                for q, order in zip(p["questions"], rank(r, docs, qs)):
                    for metric, v in question_metrics(order, chunks, q["evidence"]).items():
                        per.setdefault((r, f"{cs}/{mode}", metric), {}).setdefault(p["id"], []).append(v)

    results = []
    base = f"{BASELINE[0]}/{BASELINE[1]}"
    for (r, sysname, metric), pp in per.items():
        mean, lo, hi = cluster_bootstrap(pp)
        bp = per[(r, base, metric)]
        diff = {pid: [a - b for a, b in zip(pp[pid], bp[pid])] for pid in pp}
        dmean, dlo, dhi = cluster_bootstrap(diff)
        results.append({"retriever": r, "system": sysname, "metric": metric, "mean": mean,
                        "ci": [lo, hi], "diff_vs_struct": dmean, "diff_ci": [dlo, dhi]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print_table(results, "rec@512t")
    print_table(results, "hit@5")


def print_table(results: list[dict], metric: str) -> None:
    rows = [r for r in results if r["metric"] == metric]
    rets = sorted({r["retriever"] for r in rows})
    syss = list(dict.fromkeys(r["system"] for r in rows))
    print(f"\n{metric}  (mean [95% CI]; * = CI of difference vs struct/text excludes 0)")
    print(f"{'system':22}" + "".join(f"{r:>26}" for r in rets))
    for s in syss:
        line = f"{s:22}"
        for r in rets:
            x = next(v for v in rows if v["system"] == s and v["retriever"] == r)
            sig = "*" if x["system"] != "struct/text" and (x["diff_ci"][0] > 0 or x["diff_ci"][1] < 0) else " "
            line += f"   {x['mean']:.3f} [{x['ci'][0]:.2f},{x['ci'][1]:.2f}]{sig}"
        print(line)


if __name__ == "__main__":
    main()
