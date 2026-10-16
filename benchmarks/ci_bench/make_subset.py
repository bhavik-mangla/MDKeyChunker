"""Freeze the tiny Qasper subset used by the CI benchmark (run once; output is committed).

Reads the v4 study's data directory (MDK_STUDY_DATA=.../data/v2) and writes qasper_ci.json:
10 papers drawn with a fixed seed from the study's 300 Qasper papers, restricted to papers
with at least 3 questions and under 40,000 characters of Markdown (to keep the file small).
Each paper keeps its Markdown, its questions with gold evidence, and the study's own frozen
chunk sets (`struct` with section paths, `fixed512`) so retrieval parity with the study can
be checked without rerunning the study's chunker.

    MDK_STUDY_DATA=/path/to/data/v2 python benchmarks/ci_bench/make_subset.py
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

OUT = Path(__file__).parent / "qasper_ci.json"
SEED = 2026
N_PAPERS = 10


def main() -> None:
    data = Path(os.environ["MDK_STUDY_DATA"])
    papers = json.loads((data / "qasper_v2.json").read_text())

    def has_cache(pid: str) -> bool:
        return all((data / "cache" / "base" / v / f"{pid}.json").exists()
                   for v in ("struct", "fixed512"))

    pool = sorted((p for p in papers
                   if sum(bool(q["evidence"]) for q in p["questions"]) >= 3
                   and len(p["markdown"]) < 40_000 and has_cache(p["id"])), key=lambda p: p["id"])
    chosen = sorted(random.Random(SEED).sample(pool, N_PAPERS), key=lambda p: p["id"])
    out = []
    for p in chosen:
        sets = {}
        for v in ("struct", "fixed512"):
            path = data / "cache" / "base" / v / f"{p['id']}.json"
            raw = json.loads(path.read_text())["chunks"]
            sets[v] = [{"text": c["text"], "section": c.get("section", "")} for c in raw]
        out.append({"id": p["id"], "title": p["title"], "markdown": p["markdown"],
                    "questions": [{"qid": q["qid"], "question": q["question"],
                                   "evidence": q["evidence"]}
                                  for q in p["questions"] if q["evidence"]],
                    "study_chunks": sets})
    body = {"source": "Qasper (Dasigi et al., 2021), CC BY 4.0; "
                      "subset of the v4 study's 300 papers",
            "seed": SEED, "pool_size": len(pool), "papers": out}
    OUT.write_text(json.dumps(body, ensure_ascii=False, indent=0) + "\n")
    nq = sum(len(p["questions"]) for p in out)
    print(f"pool={len(pool)} papers={len(out)} questions={nq} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
