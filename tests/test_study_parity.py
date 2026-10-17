"""Parity with the v4 study harness on the study's own cached chunks (opt-in).

Set MDK_STUDY_DATA to the study's data/v2 directory (with cache/ and qasper_v2.json) and
install the harness dependencies, e.g.:

    MDK_STUDY_DATA=/path/to/data/v2 uv run --with pytest --with tiktoken --with numpy \
        --with requests --with rank-bm25 pytest tests/test_study_parity.py
"""
import json
import os
import random
import sys
from pathlib import Path

import pytest

from mdkeychunker.evaluate import evidence_precision, evidence_recall
from mdkeychunker.models import Chunk

DATA = os.environ.get("MDK_STUDY_DATA", "")
STUDY = Path(__file__).resolve().parents[1] / "benchmarks" / "chunking_study"
pytestmark = pytest.mark.skipif(not DATA, reason="set MDK_STUDY_DATA to run study parity tests")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def study():
    for mod in ("tiktoken", "numpy", "requests", "rank_bm25"):
        pytest.importorskip(mod)
    sys.path.insert(0, str(STUDY))
    try:
        import evaluate as study_eval  # the harness module, not mdkeychunker.evaluate
        import evaluate_corpus as study_corpus
    finally:
        sys.path.remove(str(STUDY))
    return study_eval, study_corpus


def _chunks(path: Path) -> tuple[list[dict], list[Chunk]]:
    raw = _json(path)["chunks"]
    return raw, [Chunk(text=c["text"], chunk_id=str(i)) for i, c in enumerate(raw)]


def test_qasper_recall_matches_harness(study):
    study_eval, _ = study
    data = Path(DATA)
    papers = _json(data / "qasper_v2.json")
    rng = random.Random(0)
    checked = 0
    for p in papers[:25]:
        path = data / "cache" / "base" / "struct" / f"{p['id']}.json"
        if not path.exists():
            continue
        raw, chunks = _chunks(path)
        for q in p["questions"]:
            if not q["evidence"]:
                continue
            order = list(range(len(raw)))
            rng.shuffle(order)
            want = study_eval.question_metrics(order, raw, q["evidence"])
            ranked = [chunks[i] for i in order]
            for b in (256, 512, 1024):
                got = evidence_recall(ranked, q["evidence"], budget_tokens=b,
                                      tokenizer="cl100k_base")
                assert got == want[f"rec@{b}t"], (p["id"], q["qid"], b)
            for k in (1, 3, 5):
                got = evidence_recall(ranked, q["evidence"], budget_tokens=None, k=k)
                assert got == want[f"rec@{k}"], (p["id"], q["qid"], k)
            checked += 1
    assert checked >= 50


def test_freshstack_precision_matches_harness(study):
    study_eval, study_corpus = study
    data = Path(DATA)
    sel = _json(data / "selection.json")["freshstack"]["laravel"]
    qs = [q for q in _json(data / "freshstack_laravel.json")["questions"]
          if q["qid"] in set(sel["qids"])][:20]
    assert qs, "no selected FreshStack questions"
    raw: list[dict] = []
    for d in sel["doc_ids"][:40]:
        path = data / "cache" / "base" / "struct" / f"{d}.json"
        if path.exists():
            raw += _json(path)["chunks"]
    chunks = [Chunk(text=c["text"], chunk_id=str(i)) for i, c in enumerate(raw)]
    assert raw, "no FreshStack chunks found in the cache"
    rng = random.Random(1)
    for q in qs:
        gold_texts = [e["text"] for e in q["evidence"]]
        gold = set().union(*(study_eval.ngrams(study_eval.words(t)) for t in gold_texts))
        order = list(range(len(raw)))
        rng.shuffle(order)
        want = study_corpus.question_metrics(order, raw, gold)
        got = evidence_precision([chunks[i] for i in order], gold_texts, budget_tokens=1024,
                                 tokenizer="cl100k_base")
        assert got == pytest.approx(want["prec@1024t"], abs=1e-12), q["qid"]
