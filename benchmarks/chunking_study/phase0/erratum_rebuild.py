"""Pilot erratum: rebuild the LLM variants of the docs in data/cache_fixed/REBUILD_LIST.json.

Same code and settings as the v3 pilot build (build.build_paper), with CACHE pointed at
data/cache_fixed and the document text from data/freshstack_laravel_fixed.json (commit 724c31cc).
Run:  OLLAMA_URL=http://127.0.0.1:11435 NUM_CTX=32768 .venv/bin/python phase0/erratum_rebuild.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build  # noqa: E402
from mdkeychunker.config import Config  # noqa: E402

build.CACHE = ROOT / "data" / "cache_fixed"
spec = json.loads((build.CACHE / "REBUILD_LIST.json").read_text())
assert build.NUM_CTX == spec["settings"]["NUM_CTX"], "set NUM_CTX as in REBUILD_LIST settings"
docs = {d["path"]: d["markdown"] for d in json.loads((ROOT / "data" / "freshstack_laravel_fixed.json").read_text())["docs"]}
cfg = Config(llm_provider="openai_compatible", llm_base_url=f"{build.OLLAMA}/v1",
             llm_model=spec["settings"]["model"], log_level="WARNING")
for d in spec["docs"]:
    variants = set(d["rebuild"])
    build.build_paper({"id": d["doc_id"], "markdown": docs[d["path"]]}, cfg, variants)
    for v in variants:
        p = build.CACHE / v / f"{d['doc_id']}.json"
        print(d["doc_id"], v, "ok" if p.exists() else "MISSING", flush=True)
