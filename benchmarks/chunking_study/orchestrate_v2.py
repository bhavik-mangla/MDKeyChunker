"""Unattended v2 pipeline: select -> local builds -> strong builds -> evaluate -> e2e.

Like orchestrate.py: every stage is idempotent (caches per doc), a watchdog keeps
the Ollama server on :11435 alive, builds are re-run while documents are
incomplete (bounded retries), stalls are flagged, and only actionable lines go
to data/v2/events.log. Paid spend from results/spend_ledger.jsonl is copied into
the events log at every stage boundary and whenever it moves by >= $0.05.

    .venv/bin/python orchestrate_v2.py --estimate            # token/time/$ estimate only
    .venv/bin/python orchestrate_v2.py --skip strong,e2e     # any stage or group skippable
    .venv/bin/python orchestrate_v2.py --strong-reduction 1  # plan sec. 10 reductions (1 or 2)

Stage names: select, local-qasper, local-freshstack, strong-qasper, strong-freshstack,
evaluate, e2e. Groups: local, strong.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import requests

import build_v2
import llm_api

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
EVENTS = V2 / "events.log"
LOGS = V2 / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
OLLAMA = build_v2.OLLAMA_URL
OLLAMA_LOG = V2 / "ollama.log"
LEDGER = llm_api.DEFAULT_LEDGER
STALL_MIN = 25
MAX_RETRIES = 3
POLL_S = 60
SPEND_STEP = 0.05
EXIT_BUDGET = 3
LOCAL_VARIANTS = "fixed512,fixedtok,struct,enr,cr"
STRONG_VARIANTS = "enr,cr"


def stages(strong_reduction: int = 0) -> list[dict]:
    """Pipeline stages. strong_reduction applies ANALYSIS_PLAN_v2 sec. 10 in order:
    1 = CR-strong on a random 150 Qasper papers; 2 = also E-strong on FreshStack Laravel only.
    CR-strong is not run on FreshStack (plan sec. 2)."""
    q_cr = ["--sample", "150"] if strong_reduction >= 1 else []
    fs_enr = ["--domains", "laravel"] if strong_reduction >= 2 else []
    local_extra = ["--parapack", "--fixedlen"]
    out = [
        {"name": "select", "group": "select", "cmd": [PY, "select_v2.py"]},
        {"name": "local-qasper", "group": "local", "build": ["--dataset", "qasper", "--model-tag", "local",
                                                            "--variants", LOCAL_VARIANTS, *local_extra,
                                                            "--workers", "8"]},
        {"name": "local-freshstack", "group": "local", "build": ["--dataset", "freshstack", "--model-tag", "local",
                                                                "--variants", LOCAL_VARIANTS, *local_extra,
                                                                "--workers", "4"]},
        {"name": "strong-qasper-enr", "group": "strong", "build": ["--dataset", "qasper", "--model-tag", "strong",
                                                                  "--variants", "enr", "--workers", "8"]},
        {"name": "strong-freshstack-enr", "group": "strong", "build": ["--dataset", "freshstack", "--model-tag",
                                                                      "strong", "--variants", "enr", "--workers", "8",
                                                                      *fs_enr]},
        {"name": "strong-qasper-cr", "group": "strong", "build": ["--dataset", "qasper", "--model-tag", "strong",
                                                                 "--variants", "cr", "--workers", "8", *q_cr]},
        {"name": "evaluate", "group": "evaluate", "cmd": [PY, "evaluate_v2.py"], "ollama": True},
        {"name": "e2e", "group": "e2e", "cmd": [PY, "e2e_v2.py"], "ollama": True},
    ]
    for s in out:
        if "build" in s:
            s["ollama"] = "local" in s["build"]
    return out


# --------------------------------------------------------------------------- events, ollama, spend

def event(msg: str, path: Path | None = None) -> None:
    path = path or EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{time.strftime('%m-%d %H:%M')} {msg}\n")


def ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/version", timeout=5).ok
    except requests.RequestException:
        return False


def ensure_ollama() -> bool:
    if ollama_up():
        return True
    event("WARN ollama :11435 down; restarting")
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11435", "OLLAMA_NUM_PARALLEL": os.environ.get("V2_NUM_PARALLEL", "8"),
           "OLLAMA_MAX_LOADED_MODELS": "3"}
    OLLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OLLAMA_LOG.open("a") as log:
        subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log, start_new_session=True)
    for _ in range(30):
        time.sleep(2)
        if ollama_up():
            event("OK ollama restarted")
            return True
    event("ERROR ollama failed to restart")
    return False


class SpendMonitor:
    def __init__(self, path: Path = LEDGER):
        self.ledger = llm_api.Ledger(path)
        self.last = None

    def check(self, force: bool = False) -> float:
        total = self.ledger.total_spend()
        if force or self.last is None or total - self.last >= SPEND_STEP:
            event(f"SPEND ${total:.4f} of ${llm_api.DEFAULT_BUDGET_USD:.2f} cap (ledger)")
            self.last = total
        return total


# --------------------------------------------------------------------------- builds

def _arg(build: list[str], flag: str, default: str | None = None) -> str | None:
    return build[build.index(flag) + 1] if flag in build else default


def build_targets(build: list[str], selection: dict) -> tuple[list[str], list[str], str]:
    """(doc ids, variants, model tag) the build stage must complete."""
    ds, tag = _arg(build, "--dataset"), _arg(build, "--model-tag", "local")
    variants = _arg(build, "--variants", "").split(",")
    variants += [v for v in ("parapack", "fixedlen") if f"--{v}" in build]
    if ds == "qasper":
        ids = list(selection["qasper"]["paper_ids"])
    else:
        doms = _arg(build, "--domains", "laravel,angular,yolo").split(",")
        ids = [i for d in doms for i in selection["freshstack"][d]["doc_ids"]]
    n = int(_arg(build, "--sample", "0"))
    if n:
        ids = sorted(random.Random(2027).sample(ids, min(n, len(ids))))
    return ids, variants, tag


def newest_mtime(root: Path) -> float:
    return max((p.stat().st_mtime for p in root.rglob("*.json")), default=0.0) if root.exists() else 0.0


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def run_build(stage: dict, selection: dict, spend: SpendMonitor) -> str:
    """Returns "done", "incomplete" or "budget"."""
    ids, variants, tag = build_targets(stage["build"], selection)
    log_path = LOGS / f"{stage['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = "build_v2.py " + " ".join(stage["build"])
    for attempt in range(MAX_RETRIES + 1):
        todo = build_v2.missing(ids, variants, tag)
        if not todo:
            event(f"DONE {stage['name']}")
            return "done"
        if attempt == MAX_RETRIES:
            break
        if running(pattern):
            event(f"ADOPT running {stage['name']}: {len(todo)} docs incomplete")
            proc = None
        else:
            if stage.get("ollama"):
                ensure_ollama()
            event(f"START {stage['name']} attempt {attempt + 1}: {len(todo)} docs incomplete")
            env = {**os.environ, "OLLAMA_URL": OLLAMA}
            with log_path.open("a") as log:
                proc = subprocess.Popen([PY, "build_v2.py", *stage["build"]], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_t, last_n, stalled = time.time(), len(todo), False
        while (proc.poll() is None) if proc else running(pattern):
            time.sleep(POLL_S)
            if stage.get("ollama"):
                ensure_ollama()
            spend.check()
            n = len(build_v2.missing(ids, variants, tag))
            if n < last_n or newest_mtime(build_v2.CACHE_V2 / tag) > last_t:
                last_t, last_n, stalled = time.time(), n, False
            elif not stalled and time.time() - last_t > STALL_MIN * 60:
                stalled = True
                event(f"STALL {stage['name']}: no cache writes for {STALL_MIN} min ({n} incomplete)")
        rc = proc.returncode if proc else None
        text = log_path.read_text() if log_path.exists() else ""
        spend.check(force=True)
        if rc == 2:
            event(f"REFUSED {stage['name']}: frozen config mismatch (see {log_path.name})")
            return "incomplete"
        if rc == EXIT_BUDGET:
            event(f"BUDGET_STOP {stage['name']}: llm_api budget guard stopped the run")
            return "budget"
        event(f"EXIT {stage['name']} attempt {attempt + 1} rc={rc}: "
              f"{len(build_v2.missing(ids, variants, tag))} incomplete, "
              f"{text.count(' FAILED:')} FAILED lines in log so far")
    event(f"GIVEUP {stage['name']}: {len(build_v2.missing(ids, variants, tag))} docs still incomplete")
    return "incomplete"


def run_cmd(stage: dict) -> bool:
    if stage.get("ollama") and not ensure_ollama():
        return False
    event(f"START {stage['name']}")
    log_path = LOGS / f"{stage['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OLLAMA_URL": OLLAMA}
    with log_path.open("w") as log:
        rc = subprocess.run(stage["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    event(f"{'DONE' if rc == 0 else 'ERROR'} {stage['name']} (rc={rc}) -> {log_path.relative_to(ROOT)}")
    return rc == 0


def skipped(stage: dict, skip: set, only: set) -> bool:
    if only and stage["name"] not in only and stage["group"] not in only:
        return True
    return stage["name"] in skip or stage["group"] in skip


# --------------------------------------------------------------------------- estimate

# Measured on the v3 pilot (data/cache/{enr_rk,cr} stats, 54 docs, 1,457 chunks, 4
# workers sharing one Ollama with OLLAMA_NUM_PARALLEL=4): thread-seconds per call.
V3_ENR_THREAD_S_PER_CALL = 37.2
V3_CR_THREAD_S = (0.0194, 9.60, 17.4)       # per doc: a*doc_tokens + b*calls + c
V3_WORKERS = 4
V3_ENR_IN_PER_CALL, V3_ENR_OUT_PER_CALL = 1034, 178
# OpenRouter price of openai/gpt-oss-120b @ deepinfra/bf16 (public endpoints API, 2026-10-06);
# it matches the smoke-test ledger row exactly. No cache-read discount is listed for it.
PRICE_IN, PRICE_OUT = 0.037e-6, 0.17e-6
REASONING_OUT = {"enr": 150, "cr": 80}       # assumed extra reasoning tokens per call at effort=low
CR_OUT = 55                                  # v3 mean CR output tokens per call (77,528 / 1,457)
GEN_S_PER_CALL = 2.0                         # assumed local answer wall-s per call at 4 workers
EMBED_PER_S = 40                             # assumed local embeddings per second


def estimate(selection: dict, strong_reduction: int = 0) -> dict:
    from build import CR_PROMPT
    from mdkeychunker.enricher import ENRICH_PROMPT
    enc = build_v2.ENC
    t_enr = len(enc.encode(ENRICH_PROMPT))
    t_cr = len(enc.encode(CR_PROMPT))
    ds = selection["doc_stats"]
    fs_ids = {d: selection["freshstack"][d]["doc_ids"] for d in ("laravel", "angular", "yolo")}
    sets = {"qasper": selection["qasper"]["paper_ids"], "freshstack": [i for v in fs_ids.values() for i in v]}
    out: dict = {}
    for name, ids in sets.items():
        chunks = sum(ds[i]["struct_chunks"] for i in ids)
        ctoks = sum(sum(ds[i]["struct_chunk_tokens"]) for i in ids)
        a, b, c = V3_CR_THREAD_S
        cr_thread = sum(a * ds[i]["tokens"] + b * ds[i]["struct_chunks"] + c for i in ids)
        enr_in = chunks * (t_enr + 250) + ctoks          # template + rolling keys/summary + chunk
        cr_in = sum(ds[i]["struct_chunks"] * (ds[i]["tokens"] + t_cr) for i in ids) + ctoks
        out[name] = {
            "chunks": chunks,
            "local_enr_hours": chunks * V3_ENR_THREAD_S_PER_CALL / V3_WORKERS / 3600,
            "local_cr_hours": cr_thread / V3_WORKERS / 3600,
            "strong_enr_in_tokens": enr_in,
            "strong_enr_usd": enr_in * PRICE_IN + chunks * (V3_ENR_OUT_PER_CALL + REASONING_OUT["enr"]) * PRICE_OUT,
            "strong_cr_in_tokens": cr_in,
            "strong_cr_usd": cr_in * PRICE_IN + chunks * (CR_OUT + REASONING_OUT["cr"]) * PRICE_OUT,
        }
    # Pre-declared reductions (plan sec. 10).
    if strong_reduction >= 1:
        lar = fs_ids["laravel"]
        cr_in = sum(ds[i]["struct_chunks"] * (ds[i]["tokens"] + t_cr) for i in lar)
        n = sum(ds[i]["struct_chunks"] for i in lar)
        out["freshstack"]["strong_cr_usd"] = cr_in * PRICE_IN + n * (CR_OUT + REASONING_OUT["cr"]) * PRICE_OUT
    if strong_reduction >= 2:
        out["qasper"]["strong_cr_usd"] *= 150 / len(sets["qasper"])
    nq = len(selection["qasper"]["qids"])
    out["e2e_local_hours"] = 8 * nq * GEN_S_PER_CALL / 3600
    n_index = sum(v["chunks"] for k, v in out.items() if isinstance(v, dict)) * 10
    out["evaluate_local_hours"] = 2 * n_index / EMBED_PER_S / 3600   # nomic + mxbai, ~10 indexed renderings
    return out


def print_estimate(e: dict) -> None:
    tot_usd = 0.0
    for ds in ("qasper", "freshstack"):
        v = e[ds]
        tot_usd += v["strong_enr_usd"] + v["strong_cr_usd"]
        print(f"{ds:10} chunks={v['chunks']:,}  local: enr {v['local_enr_hours']:.1f} h, cr {v['local_cr_hours']:.1f} h"
              f"  | strong: enr {v['strong_enr_in_tokens'] / 1e6:.1f}M in ${v['strong_enr_usd']:.2f}, "
              f"cr {v['strong_cr_in_tokens'] / 1e6:.1f}M in ${v['strong_cr_usd']:.2f}")
    print(f"evaluate ~{e['evaluate_local_hours']:.1f} h local embeddings; e2e ~{e['e2e_local_hours']:.1f} h local generation")
    print(f"paid total ${tot_usd:.2f} (cap ${llm_api.DEFAULT_BUDGET_USD:.2f})")


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma list of stage names or groups")
    ap.add_argument("--only", default="", help="comma list of stage names or groups")
    ap.add_argument("--strong-reduction", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    skip = {s for s in args.skip.split(",") if s}
    only = {s for s in args.only.split(",") if s}
    if args.estimate:
        print_estimate(estimate(json.loads((V2 / "selection.json").read_text()), args.strong_reduction))
        return 0
    spend = SpendMonitor()
    event("orchestrator v2 started")
    spend.check(force=True)
    budget_hit = False
    for st in stages(args.strong_reduction):
        if skipped(st, skip, only):
            event(f"SKIP {st['name']}")
            continue
        if "build" in st:
            if budget_hit and st["group"] == "strong":
                event(f"SKIP {st['name']} (budget cap reached earlier)")
                continue
            sel = json.loads((V2 / "selection.json").read_text())
            status = run_build(st, sel, spend)
            budget_hit |= status == "budget"
            if status == "incomplete":
                event(f"CONTINUE past incomplete {st['name']} (evaluation uses complete docs only)")
        else:
            ok = run_cmd(st)
            if not ok and st["name"] == "select":
                event("ABORT selection failed")
                return 1
        spend.check(force=True)
    event("orchestrator v2 finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
