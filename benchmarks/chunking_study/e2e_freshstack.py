"""FreshStack answer-level outcome (ANALYSIS_PLAN_v2 Amendment 1, sec. B; SECONDARY).

Runs after the v2 builds. For every selected FreshStack v2 question and system
(F512c, F256t, S, S+T, E, CR; plus E-strong when its cache covers a whole domain):
rank the domain's pooled chunks with the hybrid retriever (evaluate_v2.rank_all),
give the fixed local generator (qwen2.5:7b, Ollama :11435) exactly 1,024 cl100k
tokens of retrieved source text (answer.generate_answer), then label every nugget
with the judge (meta-llama/llama-3.3-70b-instruct on OpenRouter pinned to
DeepInfra "deepinfra/turbo", fp8, verified 2026-10-06 from
https://openrouter.ai/api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints).

Outcome: strict nugget recall (support only); partial recall also reported.
Acceptance (amendment B / plan sec. 8): positive control (own Stack Overflow
answer) >= 0.85 strict recall, negative control (another question's answer)
<= 0.05; with a completed author spot-check, Cohen's kappa >= 0.6 is also
required, otherwise results are labelled exploratory.

Caches (resumable):
  data/v2/e2e_fs_answers/<gen_tag>/<system>/<qid>.json   one answer per file
  data/v2/e2e_fs_judge/<sha256>.json                     one judge result per
      (judge spec, question, answer, nuggets); controls share this cache.

Outputs (results/v2/extra/):
  freshstack_e2e.json                 means, paired CIs, TOST (+/-0.05), controls,
                                      length bias, acceptance, run metadata
  freshstack_e2e_per_question.json    evaluate_v2 dump format (strata = domain)
  freshstack_e2e_rows.jsonl           every answer with its nugget labels
  spotcheck_sheet.csv / spotcheck_key.csv   blind author spot-check (120 items)

    .venv/bin/python e2e_freshstack.py --estimate     # judge token/$ estimate, no calls
    .venv/bin/python e2e_freshstack.py                # generate + judge + stats
    .venv/bin/python e2e_freshstack.py --score-spotcheck   # kappa once the sheet is filled
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

import answer
import build_v2
import evaluate
import evaluate_v2
import judge as judge_mod
import llm_api
import stats

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
ANS_CACHE = V2 / "e2e_fs_answers"
JUDGE_CACHE = V2 / "e2e_fs_judge"
OUT = ROOT / "results" / "v2" / "extra"
GEN_TAG = "local"
GEN_SPEC = build_v2.MODELS[GEN_TAG].spec
BUDGET = 1024
JUDGE_SPEC = "openrouter:meta-llama/llama-3.3-70b-instruct@deepinfra/turbo"
# DeepInfra price on OpenRouter (endpoints API, 2026-10-06): USD per token.
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 0.10e-6, 0.32e-6
SCALING_SPEC = "dashscope:qwen3-235b-a22b-instruct-2507"
# Every generator/enricher in the study; the judge must not share a family with any.
OTHER_MODELS = [m.spec for m in build_v2.MODELS.values()] + [GEN_SPEC, SCALING_SPEC]
FS_SYSTEMS = ("F512c", "F256t", "S", "S+T", "E", "CR", "E-strong")
PAIRS_TOST = [("E", "S+T"), ("E", "CR")]
PAIRS_EXTRA = [("S", "F512c"), ("S", "F256t"), ("E-strong", "E"), ("CR", "S+T")]
MARGIN = 0.05
SPOT_N = 120
SPOT_SEED = 2027
KAPPA_MIN = 0.6
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- io

def _read(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _write(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(p)


def answer_file(system: str, qid: str, cache: Path = ANS_CACHE, gen_tag: str = GEN_TAG) -> Path:
    return Path(cache) / gen_tag / system.replace("+", "p") / f"{qid}.json"


def _as_list(x) -> list:
    """Nuggets may be stored as a list or as its Python/JSON string form."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            import ast
            return list(ast.literal_eval(x))
    return []


def load_questions(selection: dict, v2: Path = V2) -> dict[str, list[dict]]:
    out = {}
    for d in evaluate_v2.FS_DOMAINS:
        keep = set(selection["freshstack"][d]["qids"])
        qs = json.loads((Path(v2) / f"freshstack_{d}.json").read_text())["questions"]
        out[d] = [{**q, "nuggets": _as_list(q.get("nuggets")), "domain": d} for q in qs if q["qid"] in keep]
    return out


# --------------------------------------------------------------------------- generation

def answer_domain(dom: str, qs: list[dict], system: str, doc_ids: list[str], cache_dir: Path, *,
                  ans_cache: Path = ANS_CACHE, complete_fn: Callable | None = None,
                  rank_fn: Callable = evaluate.rank, gen_spec: str = GEN_SPEC) -> list[dict]:
    """Answer every question of one domain with one system; one cache file per answer."""
    todo = [q for q in qs if not answer_file(system, q["qid"], ans_cache).exists()]
    if todo:
        chunks = evaluate_v2.load_chunks(system, doc_ids, cache_dir)
        docs = [evaluate.render(c, evaluate_v2.SYSTEMS_V2[system][2]) for c in chunks]
        orders = evaluate_v2.rank_all(docs, [q["question"] for q in todo], ["hybrid"], rank_fn)["hybrid"]
        for q, order in zip(todo, orders):
            g = answer.generate_answer("freshstack", q["question"], order, chunks, BUDGET, gen_spec,
                                       complete_fn=complete_fn, tag=f"e2e-freshstack-{system}")
            _write(answer_file(system, q["qid"], ans_cache),
                   {"qid": q["qid"], "domain": dom, "system": system, "answer": g["answer"], "raw": g["raw"],
                    "context_tokens": g["context_tokens"], "prompt_sha256": g["prompt_sha256"],
                    "in_tokens": g.get("in_tokens"), "out_tokens": g.get("out_tokens"),
                    "latency_s": g.get("latency_s"), "model": g.get("model"),
                    "finish_reason": g.get("finish_reason")})
    return [_read(answer_file(system, q["qid"], ans_cache)) for q in qs]


def jobs_for(plan: dict, systems=FS_SYSTEMS) -> list[tuple[str, str]]:
    out = []
    for dom in plan["groups"]:
        for s in systems:
            if s not in plan["systems"]:
                continue
            if s in evaluate_v2.STRONG and dom not in plan["strong_groups"].get(s, []):
                continue
            out.append((dom, s))
    return out


def generate(dqs: dict, plan: dict, cache_dir: Path, *, ans_cache: Path = ANS_CACHE, workers: int = 1,
             complete_fn: Callable | None = None, rank_fn: Callable = evaluate.rank, systems=FS_SYSTEMS,
             log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    rows, failed = [], []

    def one(job):
        dom, s = job
        try:
            r = answer_domain(dom, dqs[dom], s, plan["groups"][dom], cache_dir, ans_cache=ans_cache,
                              complete_fn=complete_fn, rank_fn=rank_fn)
        except llm_api.LLMError as e:
            with _LOCK:
                failed.append(f"{dom}/{s}")
            log(f"{dom} {s} FAILED: {llm_api._redact(str(e))[:200]}")
            return
        with _LOCK:
            rows.extend(r)
        log(f"{dom} {s}: {len(r)} answers")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, jobs_for(plan, systems)))
    return {"rows": rows, "failed": failed}


# --------------------------------------------------------------------------- judging

class CachedJudge(judge_mod.Judge):
    """judge.Judge with a content-addressed, resumable cache (one file per judgement)."""

    def __init__(self, model_spec: str, *, cache: Path = JUDGE_CACHE, **kw):
        super().__init__(model_spec, **kw)
        self.cache = Path(cache)
        self.hits = 0

    def key(self, question: str, answer_text: str, nuggets: list[str]) -> str:
        blob = json.dumps([self.model_spec, judge_mod.PROMPT, question.strip(), answer_text.strip(),
                           [n.strip() for n in nuggets]])
        return hashlib.sha256(blob.encode()).hexdigest()

    def judge_nuggets(self, question: str, answer: str, nuggets: list[str]) -> dict:
        p = self.cache / f"{self.key(question, answer, nuggets)}.json"
        hit = _read(p)
        if hit is not None:
            with _LOCK:
                self.hits += 1
            return hit
        r = super().judge_nuggets(question, answer, nuggets)
        _write(p, r)
        return r


def judge_prompt(question: str, answer_text: str, nuggets: list[str]) -> str:
    return judge_mod.PROMPT.format(question=question.strip(), answer=answer_text.strip(),
                                   nuggets="\n".join(f"{i}. {n.strip()}" for i, n in enumerate(nuggets, 1)))


def estimate_judge(items: list[tuple[str, str, list[str]]], answer_tokens: int | None = None) -> dict:
    """Token and $ estimate for judge calls. items: (question, answer, nuggets).
    If answer_tokens is given, every answer is assumed to be that long (pre-generation estimate)."""
    tin = tout = 0
    for q, a, ns in items:
        tin += len(answer.ENC.encode(judge_prompt(q, "" if answer_tokens else a, ns))) + (answer_tokens or 0)
        tout += 20 + 15 * len(ns)  # {"labels": [{"id": i, "label": "..."}...]}
    tin, tout = int(tin * 1.1), int(tout * 1.1)  # chat template + tokenizer mismatch (Llama vs cl100k)
    return {"calls": len(items), "in_tokens": tin, "out_tokens": tout,
            "usd": tin * JUDGE_PRICE_IN + tout * JUDGE_PRICE_OUT}


def control_items(dqs: dict, seed: int = SPOT_SEED) -> list[dict]:
    """Questions with a gold Stack Overflow answer, in a fixed random order (the
    negative control pairs each question with the next one's answer)."""
    items = [{"qid": q["qid"], "question": q["question"], "gold_answer": q["answer"], "nuggets": q["nuggets"]}
             for qs in dqs.values() for q in qs if (q.get("answer") or "").strip() and q["nuggets"]]
    items.sort(key=lambda x: x["qid"])
    random.Random(seed).shuffle(items)
    return items


def judge_rows(jg: judge_mod.Judge, rows: list[dict], qmap: dict, workers: int = 8,
               log: Callable[[str], None] = lambda s: print(s, flush=True)) -> list[dict]:
    out, failed = [], []

    def one(r):
        q = qmap[r["qid"]]
        try:
            j = jg.judge_nuggets(q["question"], r["answer"], q["nuggets"])
        except judge_mod.JudgeParseError as e:
            # Plan v2 sec. 7: failures are counted, never dropped. An unparseable judgement
            # counts as no_support for every nugget (strict = partial = 0) and is flagged.
            with _LOCK:
                failed.append((r["system"], r["qid"]))
                out.append({**r, "labels": ["no_support"] * len(q["nuggets"]), "strict": 0.0, "partial": 0.0,
                            "parse_attempts": judge_mod.MAX_PARSE_ATTEMPTS, "judge_failed": True,
                            "answer_words": judge_mod.answer_words(r["answer"])})
            log(f"judge parse failure {r['system']} {r['qid']} (counted as no_support): {e}")
            return
        with _LOCK:
            out.append({**r, "labels": j["labels"], "strict": j["strict"], "partial": j["partial"],
                        "parse_attempts": j.get("parse_attempts"),
                        "answer_words": judge_mod.answer_words(r["answer"])})

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, rows))
    out.sort(key=lambda r: (r["system"], r["qid"]))
    return out


def run_controls(jg: judge_mod.Judge, items: list[dict], workers: int = 8) -> dict:
    """judge.freshstack_controls, with the judge calls pre-warmed in parallel
    through the cache (same calls, same order of pairing)."""
    n = len(items)
    calls = [(it["question"], it["gold_answer"], it["nuggets"]) for it in items]
    if n > 1:
        calls += [(it["question"], items[(i + 1) % n]["gold_answer"], it["nuggets"]) for i, it in enumerate(items)]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(lambda c: _safe(jg.judge_nuggets, *c), calls))
    return judge_mod.freshstack_controls(jg, items)


def _safe(fn, *a):
    try:
        return fn(*a)
    except judge_mod.JudgeParseError:
        return None


# --------------------------------------------------------------------------- statistics

def per_question(judged: list[dict]) -> dict:
    per: dict = {}
    for r in judged:
        for m in ("strict", "partial"):
            per.setdefault(("hybrid", r["system"], f"nugget_{m}"), {})[r["qid"]] = [r[m]]
    return per


def analyze(judged: list[dict], strata: dict, n: int = 10_000) -> dict:
    per = per_question(judged)
    systems = sorted({r["system"] for r in judged})
    means = {}
    for s in systems:
        means[s] = {}
        for m in ("strict", "partial"):
            pp = per[("hybrid", s, f"nugget_{m}")]
            b = stats.cluster_bootstrap(pp, n=n, seed=0, levels=(0.95,), strata={k: strata[k] for k in pp})
            means[s][m] = {"mean": b["mean"], "ci95_pct": [float(x) for x in b["percentile"][0.95]],
                           "n": b["n_obs"]}
    comps = []
    for (a, b_), margin in [(p, MARGIN) for p in PAIRS_TOST] + [(p, None) for p in PAIRS_EXTRA]:
        for m in ("strict", "partial"):
            ka, kb = ("hybrid", a, f"nugget_{m}"), ("hybrid", b_, f"nugget_{m}")
            if ka not in per or kb not in per:
                continue
            mg = margin if m == "strict" else None
            comps.append({"a": a, "b": b_, "metric": f"nugget_{m}",
                          "family": "secondary" if mg else "exploratory",
                          **evaluate_v2.compare(per[ka], per[kb], margin=mg, strata=strata, n=n)})
    return {"per": per, "means": means, "comparisons": comps}


def acceptance(controls: dict | None, kappa: dict | None) -> dict:
    if not controls:
        return {"status": "not_available"}
    ok = controls["positive_pass"] and controls["negative_pass"]
    if not ok:
        status = "failed_controls"
    elif kappa and kappa.get("n"):
        status = "secondary" if kappa["kappa"] >= KAPPA_MIN else "failed_kappa"
    else:
        status = "exploratory_no_spotcheck"
    return {"status": status, "controls_pass": ok, "kappa_min": KAPPA_MIN,
            "kappa": (kappa or {}).get("kappa"), "positive_min": judge_mod.POSITIVE_MIN,
            "negative_max": judge_mod.NEGATIVE_MAX}


def relabel(body: dict) -> None:
    """Amendment 1 B: without passing controls (and kappa), every result is exploratory."""
    status = body["acceptance"]["status"]
    if status == "secondary":
        body["label"] = "SECONDARY (ANALYSIS_PLAN_v2 Amendment 1 sec. B); unadjusted 95% CIs"
        return
    body["label"] = f"EXPLORATORY ({status}; ANALYSIS_PLAN_v2 Amendment 1 sec. B); unadjusted 95% CIs"
    for c in body.get("comparisons", []):
        c["family"] = "exploratory"


# --------------------------------------------------------------------------- spot-check

SHEET_COLS = ["item_id", "question", "answer", "nugget", "author_label"]
KEY_COLS = ["item_id", "qid", "system", "nugget_index", "judge_label"]
_SHORT = {"s": "support", "p": "partial_support", "n": "no_support", "y": "support",
          "ps": "partial_support", "ns": "no_support"}


def make_spotcheck(judged: list[dict], qmap: dict, sheet: Path, key: Path, n: int = SPOT_N,
                   seed: int = SPOT_SEED, overwrite: bool = False) -> int:
    """Blind sheet of n random (answer, nugget) items; judge labels and system only in the key."""
    sheet, key = Path(sheet), Path(key)
    if sheet.exists() and not overwrite:
        return -1  # never clobber a sheet the author may be filling
    pool = sorted(((r["system"], r["qid"], i) for r in judged if not r.get("judge_failed")
                   for i in range(len(r["labels"]))))
    rng = random.Random(seed)
    picks = rng.sample(pool, min(n, len(pool)))
    rng.shuffle(picks)
    by = {(r["system"], r["qid"]): r for r in judged}
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(sheet, "w", newline="", encoding="utf-8") as fs, open(key, "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(SHEET_COLS)
        wk.writerow(KEY_COLS)
        for k, (s, qid, i) in enumerate(picks, 1):
            r, q = by[(s, qid)], qmap[qid]
            item = f"I{k:03d}"
            ws.writerow([item, q["question"], r["answer"], q["nuggets"][i], ""])
            wk.writerow([item, qid, s, i, r["labels"][i]])
    return len(picks)


def _norm_author(x: str) -> str | None:
    s = (x or "").strip().lower()
    if not s:
        return None
    s = _SHORT.get(s, s)
    return judge_mod._norm_label(s)


def score_spotcheck(sheet: Path, key: Path) -> dict:
    """Cohen's kappa (3-class, unweighted) between author and judge on filled rows;
    also binary kappa (support vs not) and raw agreement."""
    with open(key, newline="", encoding="utf-8") as f:
        kmap = {r["item_id"]: r for r in csv.DictReader(f)}
    a, j = [], []
    with open(sheet, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = _norm_author(r.get("author_label", ""))
            if lab is None or r["item_id"] not in kmap:
                continue
            a.append(lab)
            j.append(kmap[r["item_id"]]["judge_label"])
    if not a:
        return {"n": 0, "n_items": len(kmap), "kappa": None}
    labels = list(judge_mod.LABELS)
    bin_ = lambda xs: ["support" if x == "support" else "other" for x in xs]  # noqa: E731
    return {"n": len(a), "n_items": len(kmap), "complete": len(a) == len(kmap),
            "kappa": stats.cohen_kappa(a, j, labels=labels),
            "kappa_binary_support": stats.cohen_kappa(bin_(a), bin_(j), labels=["support", "other"]),
            "agreement": float(np.mean([x == y for x, y in zip(a, j)])),
            "author_counts": {lab: a.count(lab) for lab in labels},
            "judge_counts": {lab: j.count(lab) for lab in labels}}


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(V2 / "selection.json"))
    ap.add_argument("--cache", default=str(build_v2.CACHE_V2))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--gen-workers", type=int, default=2)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--judge", default=JUDGE_SPEC)
    ap.add_argument("--estimate", action="store_true", help="judge token/$ estimate only; no LLM calls")
    ap.add_argument("--skip-judge", action="store_true", help="generate answers only")
    ap.add_argument("--score-spotcheck", action="store_true", help="kappa from a filled spot-check sheet")
    ap.add_argument("--overwrite-spotcheck", action="store_true")
    ap.add_argument("--frozen", default=str(build_v2.FROZEN))
    args = ap.parse_args(argv)
    out = Path(args.out)
    sheet, key = out / "spotcheck_sheet.csv", out / "spotcheck_key.csv"

    if args.score_spotcheck:
        k = score_spotcheck(sheet, key)
        print(json.dumps(k, indent=1))
        res_p = out / "freshstack_e2e.json"
        if res_p.exists():
            res = json.loads(res_p.read_text())
            res["spotcheck"] = k
            res["acceptance"] = acceptance(res.get("controls"), k)
            relabel(res)
            res_p.write_text(json.dumps(res, indent=1))
        return 0

    sel = json.loads(Path(args.selection).read_text())
    dqs = load_questions(sel)
    qmap = {q["qid"]: q for qs in dqs.values() for q in qs}
    strata = {q["qid"]: d for d, qs in dqs.items() for q in qs}
    cache = Path(args.cache)
    plan = evaluate_v2.plan_systems({d: sel["freshstack"][d]["doc_ids"] for d in evaluate_v2.FS_DOMAINS}, cache)
    jobs = jobs_for(plan)
    controls_in = control_items(dqs)
    n_ctrl_calls = len(controls_in) * (2 if len(controls_in) > 1 else 1)

    if args.estimate:
        # Upper bound: every system (E-strong included) on every domain, 350-token answers.
        per_sys = [(q["question"], "", q["nuggets"]) for s in FS_SYSTEMS for qs in dqs.values() for q in qs]
        ctrl = [(it["question"], it["gold_answer"], it["nuggets"]) for it in controls_in] * 2
        e_ans = estimate_judge(per_sys, answer_tokens=350)
        e_ctrl = estimate_judge(ctrl)
        spent = llm_api.Ledger().total_spend()
        tot = e_ans["usd"] + e_ctrl["usd"]
        print(json.dumps({"jobs": jobs, "answers": e_ans, "controls": e_ctrl, "total_usd": tot,
                          "ledger_spent": spent, "cap": llm_api.DEFAULT_BUDGET_USD,
                          "fits_with_1.5x_headroom": spent + 1.5 * tot <= llm_api.DEFAULT_BUDGET_USD}, indent=1))
        return 0

    try:
        build_v2.check_frozen(build_v2.current_config(), Path(args.frozen), write=False, ignore_git=True)
    except build_v2.FrozenConfigMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    build_v2.configure_llm_api("freshstack")
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    print(f"freshstack e2e: jobs={jobs} dropped_docs={len(plan['dropped'])}", flush=True)
    t0 = time.time()
    gen = generate(dqs, plan, cache, workers=args.gen_workers)
    print(f"generation: {len(gen['rows'])} answers in {time.time() - t0:.0f}s, failed={gen['failed']}", flush=True)
    if args.skip_judge:
        return 1 if gen["failed"] else 0

    est = estimate_judge([(qmap[r["qid"]]["question"], r["answer"], qmap[r["qid"]]["nuggets"])
                          for r in gen["rows"]])
    est_c = estimate_judge([(it["question"], it["gold_answer"], it["nuggets"]) for it in controls_in] * 2)
    spent = llm_api.Ledger().total_spend()
    need = est["usd"] + est_c["usd"]
    print(f"judge estimate: {est['calls'] + est_c['calls']} calls, ${need:.4f} "
          f"(ledger ${spent:.4f} of ${llm_api.DEFAULT_BUDGET_USD:.2f})", flush=True)
    if spent + 1.5 * need > llm_api.DEFAULT_BUDGET_USD:
        print("REFUSED: judge estimate x1.5 would cross the ledger cap", flush=True)
        return 4

    jg = CachedJudge(args.judge, other_models=OTHER_MODELS)
    try:
        try:
            controls = run_controls(jg, controls_in, workers=args.judge_workers)
        except judge_mod.JudgeParseError as e:
            print(f"controls failed (judge parse): {e}", flush=True)
            controls = None
        judged = judge_rows(jg, gen["rows"], qmap, workers=args.judge_workers)
    except llm_api.BudgetExceeded as e:
        print(f"BUDGET_STOP: {e}", flush=True)
        return 3
    res = analyze(judged, strata, n=args.boot)
    lb = judge_mod.length_bias([{"system": r["system"], "score": r["strict"], "answer_words": r["answer_words"]}
                                for r in judged])
    out.mkdir(parents=True, exist_ok=True)
    evaluate_v2.dump_per_question(out / "freshstack_e2e_per_question.json",
                                  {"per": res["per"], "question_ids": {q: [q] for q in strata}, "strata": strata})
    with open(out / "freshstack_e2e_rows.jsonl", "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    n_sheet = make_spotcheck(judged, qmap, sheet, key, overwrite=args.overwrite_spotcheck)
    kappa = score_spotcheck(sheet, key) if sheet.exists() and key.exists() else None
    body = {
        "label": "SECONDARY (ANALYSIS_PLAN_v2 Amendment 1 sec. B); unadjusted 95% CIs",
        "meta": {"generator": GEN_SPEC, "budget_tokens": BUDGET, "retriever": "hybrid",
                 "settings": answer.GEN_SETTINGS["freshstack"], "prompt_sha256": answer.prompt_hashes(),
                 "num_ctx": build_v2.NUM_CTX["freshstack"], "judge": args.judge,
                 "judge_prompt_sha256": hashlib.sha256(judge_mod.PROMPT.encode()).hexdigest(),
                 "judge_calls": jg.calls, "judge_cache_hits": jg.hits, "judge_parse_failures": jg.parse_failures,
                 "systems": sorted(res["means"]), "jobs": jobs, "generation_failed": gen["failed"],
                 "dropped_docs": plan["dropped"], "n_answers": len(gen["rows"]), "n_judged": len(judged),
                 "margin": MARGIN, "resamples": args.boot, "controls_n": n_ctrl_calls},
        "means": res["means"], "comparisons": res["comparisons"],
        "controls": controls, "length_bias": lb,
        "spotcheck": kappa or {"n": 0, "sheet_items": n_sheet},
        "acceptance": acceptance(controls, kappa),
    }
    relabel(body)
    body["meta"]["judge_failed_rows"] = sorted([r["system"], r["qid"]] for r in judged if r.get("judge_failed"))
    (out / "freshstack_e2e.json").write_text(json.dumps(body, indent=1))
    print(f"wrote {out / 'freshstack_e2e.json'}; acceptance={body['acceptance']['status']}", flush=True)
    return 1 if gen["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
