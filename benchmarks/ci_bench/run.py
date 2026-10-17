"""Tiny Qasper BM25 benchmark: a regression guard for CI, not a claim.

Arms are the study's frozen chunk sets for the 10 subset papers, indexed with BM25
``method="study"`` (rank-bm25 BM25Okapi, bit-identical to the study), searched within each
paper, and scored with ``evidence_recall`` within 512 tokens:

  F512c  512-character windows          (study arm F512c)
  S      header-led chunks              (study arm S)
  S+T    header-led chunks + free prefix (study arm S+T, indexed text = section path + text)

and the library's own chunkers on each paper's Markdown (regression guard for mk.chunk /
mk.window; chunking always uses the zero-dependency token estimate):

  chunk           mk.chunk(max_tokens=512, prefix=True)   (the 0.4 default)
  chunk-noprefix  mk.chunk(max_tokens=512, prefix=False)
  window          mk.window(size=256, overlap=32)

``--tokenizer core`` uses the zero-dependency chars/4 estimate; ``--tokenizer cl100k_base``
(needs tiktoken) is the study's tokenizer, and then every per-question value must equal the
study's. ``--write`` refreezes golden.json (only after an intended change; say why in the PR).

    uv run python benchmarks/ci_bench/run.py [--tokenizer core|cl100k_base] [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdkeychunker.chunking import chunk, window
from mdkeychunker.evaluate import evidence_recall
from mdkeychunker.models import Chunk
from mdkeychunker.retrieval import BM25Retriever

HERE = Path(__file__).parent

BUDGET = 512
STUDY_ARMS = ("F512c", "S", "S+T")
LIBRARY_ARMS = ("chunk", "chunk-noprefix", "window")
ARMS = STUDY_ARMS + LIBRARY_ARMS


def arm_chunks(paper: dict, arm: str) -> list[Chunk]:
    md, pid = paper["markdown"], paper["id"]
    if arm == "chunk":
        return chunk(md, doc_id=pid, max_tokens=512, prefix=True)
    if arm == "chunk-noprefix":
        return chunk(md, doc_id=pid, max_tokens=512, prefix=False)
    if arm == "window":
        return window(md, size=256, overlap=32, doc_id=pid)
    sets = paper["study_chunks"]
    raw = sets["fixed512"] if arm == "F512c" else sets["struct"]
    out = []
    for i, c in enumerate(raw):
        embed = f"{c['section']}\n\n{c['text']}" if arm == "S+T" else ""
        out.append(Chunk(text=c["text"], embed_text=embed, doc_id=paper["id"],
                         chunk_id=f"{arm}:{i}"))
    return out


def run(tokenizer: str | None) -> dict:
    papers = json.loads((HERE / "qasper_ci.json").read_text())["papers"]
    per: dict[str, dict[str, float]] = {a: {} for a in ARMS}
    for p in papers:
        for arm in ARMS:
            r = BM25Retriever(arm_chunks(p, arm), method="study")
            hits = r.search_many([q["question"] for q in p["questions"]], k=len(r))
            for q, h in zip(p["questions"], hits):
                per[arm][q["qid"]] = evidence_recall(h, q["evidence"], BUDGET, tokenizer=tokenizer)
    means = {a: sum(v.values()) / len(v) for a, v in per.items()}
    return {"budget_tokens": BUDGET, "n_papers": len(papers), "n_questions": len(per["S"]),
            "mean": means, "per_question": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="core", choices=("core", "cl100k_base"))
    ap.add_argument("--write", action="store_true", help="refreeze golden.json for this tokenizer")
    args = ap.parse_args()
    res = run(None if args.tokenizer == "core" else args.tokenizer)
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=1))
    if args.write:
        path = HERE / "golden.json"
        golden = json.loads(path.read_text()) if path.exists() else {}
        golden[args.tokenizer] = res
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
