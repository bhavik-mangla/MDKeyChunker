"""Within-family scaling arm (ANALYSIS_PLAN_v2 Amendment 1, sec. C; EXPLORATORY).

E (single-call enrichment, rolling keys) with Qwen3-235B-A22B-Instruct-2507 on
DashScope (temperature 0, thinking off, seed 0) for a random 30 of the 300 Qasper
v2 papers: random.Random(2027).sample(sorted(paper ids), 30).

Reuses build_v2's enrichment adapter (LLMAdapter: parse-failure retries with
seeds 1..3, then empty fields, flagged) and enrich(); only the model differs.
  cache          data/v2/cache/qwen235b/enr/<paper>.json
  frozen config  results/v2/frozen_config_scaling.json (results/v2/frozen_config.json
                 is never read for writing nor modified here)
Nothing is written under the base/local/strong cache tags.

Token cap: llm_api refuses a DashScope call once ledger tokens + max_tokens would
cross DASHSCOPE_TOKEN_CAP. Here every call also reserves its estimated prompt
tokens + max_tokens while in flight, so concurrent workers cannot overshoot the
cap together. Hitting the cap stops the run cleanly (exit 3); finished papers
stay cached, an unfinished paper is not cached (a rerun resumes).

    .venv/bin/python build_scaling.py                 # build (resumable)
    .venv/bin/python build_scaling.py --dry-run       # sample + token estimate only
    .venv/bin/python build_scaling.py --evaluate      # AFTER the v2 build: E vs E-strong
                                                      # vs E-qwen235b vs S+T, hybrid + bm25
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config
from mdkeychunker.enricher import ENRICH_PROMPT

import build_v2
import llm_api

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
FROZEN_SCALING = ROOT / "results" / "v2" / "frozen_config_scaling.json"
OUT = ROOT / "results" / "v2" / "extra"
TAG = "qwen235b"
SAMPLE_N = 30
SAMPLE_SEED = 2027
MODEL = build_v2.ModelConfig(TAG, "dashscope:qwen3-235b-a22b-instruct-2507", reasoning="off",
                             temperature=0.0, seed=0, enrich_max_tokens=1000)
# label -> (cache tag, variant, index rendering), as evaluate_v2.SYSTEMS_V2
EVAL_SYSTEMS = {"S+T": ("base", "struct", "tc"), "E": ("local", "enr", "meta"),
                "E-strong": ("strong", "enr", "meta"), "E-qwen235b": (TAG, "enr", "meta")}
EVAL_RETRIEVERS = ("hybrid", "bm25")
EVAL_PAIRS = [("E-qwen235b", "E"), ("E-qwen235b", "E-strong"), ("E-qwen235b", "S+T"),
              ("E-strong", "E"), ("E", "S+T")]


def sample_ids(all_ids: list[str], n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> list[str]:
    return random.Random(seed).sample(sorted(all_ids), n)


def scaling_config(paper_ids: list[str]) -> dict:
    o = MODEL.options()
    for k in ("allow_fallbacks", "data_collection", "banned", "providers"):  # OpenRouter-only fields
        o.pop(k, None)
    return {
        "plan": "ANALYSIS_PLAN_v2 Amendment 1 sec. C (exploratory within-family scaling)",
        "model": {"tag": TAG, "spec": MODEL.spec, "thinking": False, **o},
        "variant": "enr",
        "prompts_sha256": {"enrich": build_v2.sha256(ENRICH_PROMPT)},
        "chunking": {"struct": {"min_chunk_size": Config().min_chunk_size,
                                "max_chunk_size": Config().max_chunk_size}},
        "retry": {"parse_retries": build_v2.PARSE_RETRIES,
                  "retry_seeds": list(range(1, build_v2.PARSE_RETRIES + 1))},
        "sample": {"dataset": "qasper", "seed": SAMPLE_SEED, "n": SAMPLE_N,
                   "method": "random.Random(2027).sample(sorted(selection qasper paper_ids), 30)",
                   "paper_ids": paper_ids},
        "dashscope_token_cap": llm_api.DASHSCOPE_TOKEN_CAP,
        "cache": f"data/v2/cache/{TAG}/enr",
        # Recorded, not compared: committing these new files must not block a resume.
        "git": {"harness": build_v2.git_state(ROOT)},
    }


class TokenGuard:
    """complete_fn wrapper: reserve estimated prompt + max_tokens per in-flight DashScope
    call, so ledger tokens + reservations never exceed the cap."""

    def __init__(self, cap: int = llm_api.DASHSCOPE_TOKEN_CAP, complete_fn: Callable | None = None,
                 used_fn: Callable[[], int] | None = None):
        self.cap = cap
        self.inner = complete_fn or llm_api.complete
        spec = llm_api.ModelSpec.parse(MODEL.spec)
        self.used_fn = used_fn or (lambda: llm_api.default_client().ledger.tokens_for(spec.backend, spec.model))
        self.inflight = 0
        self.lock = threading.Lock()

    @staticmethod
    def estimate(prompt: str) -> int:
        return int(len(build_v2.ENC.encode(prompt)) * 1.2) + 64  # Qwen vs cl100k, chat template

    def __call__(self, spec, prompt, **kw):
        need = self.estimate(prompt) + int(kw.get("max_tokens") or 0)
        with self.lock:
            used = self.used_fn()
            if used + self.inflight + need > self.cap:
                raise llm_api.BudgetExceeded(f"dashscope token cap {self.cap}: used {used} + in flight "
                                             f"{self.inflight} + this call {need}")
            self.inflight += need
        try:
            return self.inner(spec, prompt, **kw)
        finally:
            with self.lock:
                self.inflight -= need


class ScalingBuilder(build_v2.Builder):
    """build_v2.Builder restricted to the enr variant under the qwen235b tag."""

    @property
    def model(self) -> build_v2.ModelConfig:
        return MODEL

    def build_doc(self, doc: dict) -> dict:
        if self.stop.is_set():
            raise build_v2.BudgetStop("run stopped (budget)")
        if self._get("enr", doc["id"]):
            return {}
        struct = MarkdownChunker(Config(log_level="WARNING")).chunk(doc["markdown"])
        ad = build_v2.LLMAdapter(self.model, self.complete_fn, self.stop, tag="enrich-qasper-scaling")
        chunks, st = build_v2.enrich(struct, ad)
        self._save("enr", doc["id"], chunks, st)
        return {"enr": st}


def load_sample(selection: dict) -> list[dict]:
    ids = sample_ids(selection["qasper"]["paper_ids"])
    return build_v2.load_docs("qasper", {"qasper": {"paper_ids": ids}})


def estimate_tokens(docs: list[dict]) -> dict:
    """Rough DashScope tokens: per chunk, prompt (template + chunk + ~keys) and ~200 output tokens."""
    ch = MarkdownChunker(Config(log_level="WARNING"))
    base = len(build_v2.ENC.encode(ENRICH_PROMPT))
    n = tin = 0
    for d in docs:
        for c in ch.chunk(d["markdown"]):
            n += 1
            tin += int((base + len(build_v2.ENC.encode(c.text)) + 150) * 1.1)
    return {"chunks": n, "in_tokens": tin, "out_tokens": n * 200, "total": tin + n * 200}


# --------------------------------------------------------------------------- evaluation (after the build)

def _load(label: str, pid: str, cache: Path) -> list[dict] | None:
    tag, variant, _ = EVAL_SYSTEMS[label]
    p = build_v2.cache_path(tag, variant, pid, cache)
    return json.loads(p.read_text())["chunks"] if p.exists() else None


def evaluate_scaling(papers: list[dict], cache: Path, rank_fn: Callable | None = None, n: int = 10_000) -> dict:
    """Retrieval metrics on the sampled papers that every system covers; paired, exploratory."""
    import evaluate
    import evaluate_v2
    rank_fn = rank_fn or evaluate.rank
    per: dict = {}
    used, skipped = [], []
    for p in papers:
        qs = [q for q in p["questions"] if q["evidence"]]
        loaded = {s: _load(s, p["id"], cache) for s in EVAL_SYSTEMS}
        if not qs or any(v is None for v in loaded.values()):
            skipped.append(p["id"])
            continue
        used.append(p["id"])
        for s, chunks in loaded.items():
            docs = [evaluate.render(c, EVAL_SYSTEMS[s][2]) for c in chunks]
            orders = evaluate_v2.rank_all(docs, [q["question"] for q in qs], list(EVAL_RETRIEVERS), rank_fn)
            for r in EVAL_RETRIEVERS:
                for q, order in zip(qs, orders[r]):
                    for m, v in evaluate.question_metrics(order, chunks, q["evidence"]).items():
                        per.setdefault((r, s, m), {}).setdefault(p["id"], []).append(v)
    means = {"|".join(k): float(sum(x for xs in v.values() for x in xs) / sum(len(xs) for xs in v.values()))
             for k, v in per.items() if k[2] in ("rec@512t", "rec@1024t", "rec@256t")}
    comps = []
    for r in EVAL_RETRIEVERS:
        for a, b in EVAL_PAIRS:
            for m in ("rec@512t",):
                if (r, a, m) in per and (r, b, m) in per:
                    comps.append({"retriever": r, "a": a, "b": b, "metric": m, "family": "exploratory",
                                  **evaluate_v2.compare(per[(r, a, m)], per[(r, b, m)], n=n)})
    return {"label": "EXPLORATORY (Amendment 1 sec. C); unadjusted 95% CIs", "papers_used": used,
            "papers_skipped": skipped, "means": means, "comparisons": comps,
            "per": {"|".join(k): v for k, v in per.items()}}


def build_complete() -> bool:
    """The v2 evaluation (stats.json) exists only after the main build and evaluation."""
    return (ROOT / "results" / "v2" / "stats.json").exists()


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(V2 / "selection.json"))
    ap.add_argument("--cache", default=str(build_v2.CACHE_V2))
    ap.add_argument("--frozen", default=str(FROZEN_SCALING))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N sampled papers (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="print the sample and a token estimate")
    ap.add_argument("--evaluate", action="store_true", help="retrieval comparison (only after the v2 build)")
    ap.add_argument("--force-evaluate", action="store_true", help="skip the build-complete check")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    if Path(args.frozen).resolve() == build_v2.FROZEN.resolve():
        ap.error("the scaling run must not use results/v2/frozen_config.json")
    sel = json.loads(Path(args.selection).read_text())
    docs = load_sample(sel)
    ids = [d["id"] for d in docs]
    cache = Path(args.cache)

    if args.evaluate:
        if not (build_complete() or args.force_evaluate):
            print("REFUSED: results/v2/stats.json absent; the v2 build/evaluation is not complete "
                  "(no outcome viewing before then). Use --force-evaluate to override.", flush=True)
            return 2
        import evaluate
        evaluate.OLLAMA = build_v2.OLLAMA_URL
        papers = {p["id"]: p for p in json.loads((V2 / "qasper_v2.json").read_text())}
        res = evaluate_scaling([papers[i] for i in ids], cache, n=args.boot)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "scaling_qasper.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {OUT / 'scaling_qasper.json'} ({len(res['papers_used'])} papers)", flush=True)
        return 0

    est = estimate_tokens(docs)
    spec = llm_api.ModelSpec.parse(MODEL.spec)
    used = llm_api.default_client().ledger.tokens_for(spec.backend, spec.model)
    print(f"scaling sample ({len(ids)}): {','.join(ids)}", flush=True)
    print(f"token estimate {est} ; dashscope used {used} of cap {llm_api.DASHSCOPE_TOKEN_CAP}", flush=True)
    if args.dry_run:
        return 0
    try:
        build_v2.check_frozen(scaling_config(ids), Path(args.frozen), ignore_git=True)
    except build_v2.FrozenConfigMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    if args.limit:
        docs = docs[: args.limit]
    todo = [d for d in docs if not build_v2.cache_path(TAG, "enr", d["id"], cache).exists()]
    print(f"qasper/{TAG}: {len(todo)}/{len(docs)} papers to build", flush=True)
    t0 = time.time()
    b = ScalingBuilder("qasper", TAG, ["enr"], cache=cache, complete_fn=TokenGuard())
    res = b.run(todo, args.workers)
    done = sum(build_v2.cache_path(TAG, "enr", i, cache).exists() for i in ids)
    used = llm_api.default_client().ledger.tokens_for(spec.backend, spec.model)
    print(f"SUMMARY done_this_run={res['done']} papers_complete={done}/{len(ids)} failed={len(res['failed'])} "
          f"parse_failures={res['parse_failures']} tokens_used={used}/{llm_api.DASHSCOPE_TOKEN_CAP} "
          f"budget_stop={res['budget_stop']} elapsed={time.time() - t0:.0f}s", flush=True)
    if res["budget_stop"]:
        return 3
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
