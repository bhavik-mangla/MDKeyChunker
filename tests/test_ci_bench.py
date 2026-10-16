"""Tiny Qasper BM25 benchmark (benchmarks/ci_bench): regression guard against golden.json.

The core run needs no dependency; the cl100k run (tiktoken) must reproduce the study's
per-question values, which golden.json holds (checked against the study's
results/v2/qasper_per_question.json when it was frozen: 117/117 equal).
"""
import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "ci_bench"
GOLDEN = json.loads((BENCH / "golden.json").read_text())
TOLERANCE = 0.01  # +/-1 point on the mean (RESEARCH.md sec. 4)


def _run(tokenizer):
    spec = importlib.util.spec_from_file_location("ci_bench_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(tokenizer)


def _check(res, golden):
    assert (res["n_papers"], res["n_questions"]) == (golden["n_papers"], golden["n_questions"])
    for arm, want in golden["mean"].items():
        assert abs(res["mean"][arm] - want) <= TOLERANCE, (arm, res["mean"][arm], want)
    changed = {(arm, q) for arm, vals in golden["per_question"].items()
               for q, v in vals.items() if res["per_question"][arm][q] != v}
    assert not changed, f"{len(changed)} per-question values changed, e.g. {sorted(changed)[:3]}"


def test_bm25_benchmark_core_tokenizer():
    _check(_run(None), GOLDEN["core"])


def test_bm25_benchmark_reproduces_study_with_cl100k():
    pytest.importorskip("tiktoken")
    _check(_run("cl100k_base"), GOLDEN["cl100k_base"])
