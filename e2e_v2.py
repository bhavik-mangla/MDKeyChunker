"""Qasper end-to-end answers (ANALYSIS_PLAN_v2 sec. 5, key secondary).

For each paper, system and question: rank the paper's chunks with the hybrid
retriever (as evaluate_v2), give the fixed local generator exactly 512 cl100k
tokens of retrieved source text (answer.build_context), and score the answer
with the official Qasper token F1, max over every annotator's reference
(answer.qasper_answer_f1 / qasper_references).

Systems: F512c, F256t, S, S+T, E, CR, plus E-strong / CR-strong when their
caches cover the paper. The question set is the retrieval set (questions with
text evidence) on the papers complete for every non-strong arm, so answer F1
pairs with the retrieval metrics.

Resumable: one cache file per (generator tag, system, paper) under
data/v2/e2e/<gen_tag>/<system>/<paper_id>.json, written after every answer.
Writes results/v2/qasper_e2e.json (means, run metadata) and
results/v2/qasper_e2e_per_question.json ({"hybrid|<system>|answer_f1": {paper: [f1...]}}),
then re-runs evaluate_v2.analyze() so answer F1 enters the key-secondary family.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import answer
import build_v2
import evaluate
import evaluate_v2
import llm_api

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
E2E_CACHE = V2 / "e2e"
OUT = ROOT / "results" / "v2"
GEN_TAG = "local"
GEN_SPEC = build_v2.MODELS[GEN_TAG].spec
BUDGET = 512
E2E_SYSTEMS = ("F512c", "F256t", "S", "S+T", "E", "CR", "E-strong", "CR-strong")
_LOCK = threading.Lock()


def cache_file(system: str, pid: str, cache: Path = E2E_CACHE, gen_tag: str = GEN_TAG) -> Path:
    return Path(cache) / gen_tag / system.replace("+", "p") / f"{pid}.json"


def _read(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _write(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(p)


def answer_paper(paper: dict, system: str, cache_dir: Path, *, e2e_cache: Path = E2E_CACHE,
                 complete_fn: Callable | None = None, rank_fn: Callable = evaluate.rank,
                 gen_spec: str = GEN_SPEC) -> list[dict]:
    """Answer every evidence question of one paper with one system; cached per question."""
    qs = [q for q in paper["questions"] if q["evidence"]]
    if not qs:
        return []
    cf = cache_file(system, paper["id"], e2e_cache)
    done = _read(cf)
    chunks = evaluate_v2.load_chunks(system, [paper["id"]], cache_dir)
    todo = [q for q in qs if q["qid"] not in done]
    if todo:
        docs = [evaluate.render(c, evaluate_v2.SYSTEMS_V2[system][2]) for c in chunks]
        orders = evaluate_v2.rank_all(docs, [q["question"] for q in todo], ["hybrid"], rank_fn)["hybrid"]
        for q, order in zip(todo, orders):
            g = answer.generate_answer("qasper", q["question"], order, chunks, BUDGET, gen_spec,
                                       complete_fn=complete_fn, tag=f"e2e-qasper-{system}")
            f1, typ = answer.qasper_answer_f1(g["answer"], answer.qasper_references(q["answers"]))
            done[q["qid"]] = {"qid": q["qid"], "paper": paper["id"], "system": system, "answer": g["answer"],
                              "raw": g["raw"], "f1": f1, "best_ref_type": typ,
                              "context_tokens": g["context_tokens"], "prompt_sha256": g["prompt_sha256"],
                              "in_tokens": g.get("in_tokens"), "out_tokens": g.get("out_tokens"),
                              "latency_s": g.get("latency_s"), "model": g.get("model")}
            _write(cf, done)
    return [done[q["qid"]] for q in qs]


def run(papers: list[dict], plan: dict, cache_dir: Path, *, e2e_cache: Path = E2E_CACHE, workers: int = 4,
        complete_fn: Callable | None = None, rank_fn: Callable = evaluate.rank, systems=E2E_SYSTEMS,
        log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    """Returns {"per": {(hybrid, system, answer_f1): {pid: [f1]}}, "rows": [...], "question_ids"}."""
    jobs = []
    for p in papers:
        if p["id"] not in plan["groups"]:
            continue
        for s in systems:
            if s not in plan["systems"]:
                continue
            if s in evaluate_v2.STRONG and p["id"] not in plan["strong_groups"].get(s, []):
                continue
            jobs.append((p, s))
    rows: list[dict] = []
    failed: list[str] = []

    def one(job):
        p, s = job
        try:
            r = answer_paper(p, s, cache_dir, e2e_cache=e2e_cache, complete_fn=complete_fn, rank_fn=rank_fn)
        except llm_api.LLMError as e:
            failed.append(f"{p['id']}/{s}")
            log(f"{p['id']} {s} FAILED: {llm_api._redact(str(e))[:200]}")
            return
        with _LOCK:
            rows.extend(r)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, jobs))
    order = {p["id"]: [q["qid"] for q in p["questions"] if q["evidence"]] for p in papers}
    per: dict = {}
    by = {(r["system"], r["qid"]): r for r in rows}
    for (p, s) in jobs:
        vals = [by[(s, q)]["f1"] for q in order[p["id"]] if (s, q) in by]
        if len(vals) == len(order[p["id"]]) and vals:
            per.setdefault(("hybrid", s, "answer_f1"), {})[p["id"]] = vals
    return {"per": per, "rows": rows, "failed": failed,
            "question_ids": {pid: q for pid, q in order.items() if pid in plan["groups"] and q}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(V2 / "selection.json"))
    ap.add_argument("--cache", default=str(build_v2.CACHE_V2))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N papers (smoke runs)")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--frozen", default=str(build_v2.FROZEN))
    args = ap.parse_args(argv)
    try:
        build_v2.check_frozen(build_v2.current_config(), Path(args.frozen), write=False, ignore_git=True)
    except build_v2.FrozenConfigMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    build_v2.configure_llm_api("qasper")
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    sel = json.loads(Path(args.selection).read_text())
    papers = json.loads((V2 / "qasper_v2.json").read_text())
    keep = sel["qasper"]["paper_ids"][: args.limit or None]
    papers = [p for p in papers if p["id"] in set(keep)]
    cache = Path(args.cache)
    plan = evaluate_v2.plan_systems({p["id"]: [p["id"]] for p in papers}, cache)
    res = run(papers, plan, cache, workers=args.workers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluate_v2.dump_per_question(out / "qasper_e2e_per_question.json", res, {"rows": res["rows"]})
    means = {k[1]: sum(v for vs in pp.values() for v in vs) / sum(len(vs) for vs in pp.values())
             for k, pp in res["per"].items()}
    meta = {"generator": GEN_SPEC, "budget_tokens": BUDGET, "retriever": "hybrid",
            "settings": answer.GEN_SETTINGS["qasper"], "prompt_sha256": answer.prompt_hashes(),
            "num_ctx": build_v2.NUM_CTX["qasper"], "systems": sorted(means), "failed": res["failed"],
            "dropped_docs": plan["dropped"]}
    (out / "qasper_e2e.json").write_text(json.dumps({"meta": meta, "mean_f1": means}, indent=1))
    for s, m in sorted(means.items()):
        print(f"{s:10} answer F1 {m:.4f}")
    if not args.no_stats and (out / "qasper_per_question.json").exists():
        evaluate_v2.analyze(out, n=args.boot)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
