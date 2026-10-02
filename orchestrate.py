"""Unattended pipeline: build -> evaluate for Qasper, then FreshStack.

Each stage is idempotent (build.py caches per paper/variant), so the
orchestrator can resume after any crash. A watchdog loop restarts the Ollama
server if it dies, re-runs a build while papers are incomplete (bounded
retries), and flags stalls. Only actionable lines go to data/events.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
EVENTS = DATA / "events.log"
PY = str(ROOT / ".venv" / "bin" / "python")
OLLAMA = "http://127.0.0.1:11435"
OLLAMA_LOG = Path("/private/tmp/claude-501/-Users-bhavikmangla-Developer/"
                  "954267a2-7ebc-4884-a8d4-d47aff02f41f/scratchpad/ollama2.log")
STALL_MIN = 25
MAX_RETRIES = 3

STAGES = [
    {"name": "qasper-build", "subset": "data/subset.json", "num_ctx": "16384",
     "variants": "fixed512,fixedtok,struct,enr_rk,enr_nork,merged_rk,merged_nork,cr",
     "log": "data/build.log"},
    {"name": "qasper-eval", "cmd": [PY, "evaluate.py", "--subset", "data/subset.json",
                                    "--out", "results/qasper.json"], "log": "results/qasper.txt"},
    {"name": "freshstack-build", "subset": "data/freshstack_docs.json", "num_ctx": "32768",
     "variants": "fixed512,fixedtok,struct,enr_rk,merged_rk,cr", "log": "data/build_freshstack.log"},
    {"name": "freshstack-eval", "cmd": [PY, "evaluate_corpus.py", "--out", "results/freshstack.json"],
     "log": "results/freshstack.txt"},
]


def event(msg: str) -> None:
    line = f"{time.strftime('%H:%M')} {msg}"
    with EVENTS.open("a") as f:
        f.write(line + "\n")


def ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA}/api/version", timeout=5).ok
    except requests.RequestException:
        return False


def ensure_ollama() -> None:
    if ollama_up():
        return
    event("WARN ollama :11435 down; restarting")
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11435", "OLLAMA_NUM_PARALLEL": "4",
           "OLLAMA_MAX_LOADED_MODELS": "3"}
    with OLLAMA_LOG.open("a") as log:
        subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log, start_new_session=True)
    for _ in range(30):
        time.sleep(2)
        if ollama_up():
            event("OK ollama restarted")
            return
    event("ERROR ollama failed to restart")


def missing(stage: dict) -> list[str]:
    items = json.loads((ROOT / stage["subset"]).read_text())
    variants = stage["variants"].split(",")
    return [p["id"] for p in items
            if not all((CACHE / v / f"{p['id']}.json").exists() for v in variants)]


def newest_cache_mtime() -> float:
    return max((p.stat().st_mtime for p in CACHE.rglob("*.json")), default=0.0)


def running_build(subset: str) -> bool:
    return subprocess.run(["pgrep", "-f", f"build.py --subset {subset}"],
                          capture_output=True).returncode == 0


def run_build(stage: dict) -> bool:
    for attempt in range(MAX_RETRIES + 1):
        todo = missing(stage)
        if not todo:
            event(f"DONE {stage['name']}")
            return True
        if attempt == MAX_RETRIES:
            break
        if not running_build(stage["subset"]):
            ensure_ollama()
            event(f"START {stage['name']} attempt {attempt + 1}: {len(todo)} items incomplete")
            env = {**os.environ, "OLLAMA_URL": OLLAMA, "NUM_CTX": stage["num_ctx"]}
            with (ROOT / stage["log"]).open("a") as log:
                subprocess.Popen([PY, "build.py", "--subset", stage["subset"], "--workers", "4",
                                  "--variants", stage["variants"]],
                                 cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
            time.sleep(5)
        else:
            event(f"ADOPT running {stage['name']}: {len(todo)} items incomplete")
        last_progress, last_count, stalled = time.time(), len(todo), False
        while running_build(stage["subset"]):
            time.sleep(60)
            ensure_ollama()
            n = len(missing(stage))
            if n < last_count or newest_cache_mtime() > last_progress:
                last_progress, last_count, stalled = time.time(), n, False
            elif not stalled and time.time() - last_progress > STALL_MIN * 60:
                stalled = True
                event(f"STALL {stage['name']}: no cache writes for {STALL_MIN} min ({n} incomplete)")
        failed = [l for l in (ROOT / stage["log"]).read_text().splitlines() if "FAILED" in l]
        event(f"EXIT {stage['name']} attempt {attempt + 1}: {len(missing(stage))} incomplete, "
              f"{len(failed)} FAILED lines in log so far")
    event(f"GIVEUP {stage['name']}: still incomplete {missing(stage)}")
    return False


def run_eval(stage: dict) -> bool:
    ensure_ollama()
    event(f"START {stage['name']}")
    env = {**os.environ, "OLLAMA_URL": OLLAMA}
    with (ROOT / stage["log"]).open("w") as log:
        rc = subprocess.run(stage["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    event(f"{'DONE' if rc == 0 else 'ERROR'} {stage['name']} (rc={rc}) -> {stage['log']}")
    return rc == 0


def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    event("orchestrator started")
    for stage in STAGES:
        ok = run_build(stage) if "subset" in stage else run_eval(stage)
        if not ok and "subset" in stage:
            event(f"CONTINUE past incomplete {stage['name']} (evaluation uses complete items only)")
    event("orchestrator finished")


if __name__ == "__main__":
    main()
