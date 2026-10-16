"""Diagnostic for the late-chunking arm: mean within-document cosine between chunk embeddings,
late-chunked (LC-S) against plain jina-embeddings-v3 (S). Reads the caches only; no model calls.
Writes results/v2/extra/lc_similarity.json."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

import build_v2
import evaluate
import evaluate_v2 as ev
import retrieval_extra as rx

OUT = Path(__file__).parent / "results" / "v2" / "extra" / "lc_similarity.json"


def mean_offdiag_cos(M: np.ndarray) -> float:
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    n = len(M)
    return float(((M @ M.T).sum() - n) / (n * n - n))


def main() -> None:
    res: dict = {}
    for f in sorted(rx.LC_CACHE.glob("*.npz")):
        doc = re.sub(r"\.[0-9a-f]{16}\.npz$", "", f.name)
        ds = "freshstack" if "__" in doc else "qasper"
        late = np.load(f)["emb"]
        chunks = ev.load_chunks("S", [doc], build_v2.CACHE_V2)
        texts = [evaluate.render(c, ev.SYSTEMS_V2["S"][2]) for c in chunks]
        paths = [rx.JINA_CACHE / (hashlib.sha1(f"{rx.JinaV3.name}|retrieval.passage|{t}".encode()).hexdigest() + ".npy")
                 for t in texts]
        if len(texts) < 2 or len(late) != len(texts) or not all(p.exists() for p in paths):
            res.setdefault(ds, {"skipped": 0}).setdefault("skipped", 0)
            res[ds]["skipped"] += 1
            continue
        plain = np.vstack([np.load(p) for p in paths])
        r = res.setdefault(ds, {"skipped": 0})
        r.setdefault("plain", []).append(mean_offdiag_cos(plain))
        r.setdefault("late", []).append(mean_offdiag_cos(late))
    out = {ds: {"n_docs": len(r.get("plain", [])), "skipped": r["skipped"],
                "plain": float(np.mean(r["plain"])), "late": float(np.mean(r["late"]))}
           for ds, r in res.items()}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
