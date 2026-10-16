"""Erratum (v3 PILOT, FreshStack-Laravel): re-run evaluate_corpus.py with a configurable
cache root and data file, without editing evaluate_corpus.py.

  --cache   chunk-set root (default data/cache_fixed); evaluate_corpus only evaluates
            systems whose chunk set exists for every doc, so a root holding only
            fixed512/fixedtok/struct yields the zero-LLM systems
  --data    dataset (default data/freshstack_laravel_fixed.json)
  --out     results file (a *_per_question.json is written next to it)

Embeddings go through phase0/common.embed: vectors are read from data/emb (read-only) or
data/emb_phase0, and only texts never embedded before are sent to Ollama. Set
OLLAMA_URL=http://127.0.0.1:11434 to pin the server.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402  (sets sys.path to the repo root)
import evaluate_corpus  # noqa: E402

ROOT = common.ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_fixed"))
    ap.add_argument("--data", default=str(ROOT / "data" / "freshstack_laravel_fixed.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase0" / "erratum_freshstack.json"))
    ap.add_argument("--retrievers", default="bm25,mxbai,nomic,hybrid")
    args = ap.parse_args()

    evaluate_corpus.CACHE = Path(args.cache)
    evaluate_corpus.rank = common.rank  # same BM25 / RRF; cache-aware embeddings
    sys.argv = [sys.argv[0], "--data", args.data, "--out", args.out, "--retrievers", args.retrievers]
    evaluate_corpus.main()


if __name__ == "__main__":
    main()
