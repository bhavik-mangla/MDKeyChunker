"""diagnose reproduces a paper finding end to end (opt-in: needs the study's data/v2).

Exploratory row of results/v2/stats.json: Qasper, BM25, enrichment (E, 7B LLM) minus free
prefix (S+T), recall within 512 cl100k tokens: +0.087 [0.0665, 0.1085] over 300 papers.

    MDK_STUDY_DATA=/path/to/data/v2 uv run --with pytest --with numpy --with tiktoken \
        pytest tests/test_diagnose_study.py
"""
import json
import os
from pathlib import Path

import pytest

from mdkeychunker.models import Chunk

DATA = os.environ.get("MDK_STUDY_DATA", "")
RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "chunking_study" / "results" / "v2"
pytestmark = pytest.mark.skipif(not DATA, reason="set MDK_STUDY_DATA to run study parity tests")


def render_meta(c: dict) -> str:  # the study's evaluate.render(c, "meta")
    head = [c.get("title", ""), c.get("summary", ""), ", ".join(c.get("keywords", [])),
            " ".join(c.get("questions", []))]
    return "\n".join(h for h in head if h) + "\n\n" + c["text"]


def test_diagnose_reproduces_bm25_enrichment_gain_on_qasper():
    pytest.importorskip("numpy")
    pytest.importorskip("tiktoken")
    from mdkeychunker.diagnose import diagnose

    data = Path(DATA)
    sel = set(json.loads((data / "selection.json").read_text())["qasper"]["paper_ids"])
    base, enr, qa = [], [], []
    for p in json.loads((data / "qasper_v2.json").read_text()):
        if p["id"] not in sel:
            continue
        s = json.loads((data / "cache/base/struct" / f"{p['id']}.json").read_text())["chunks"]
        e = json.loads((data / "cache/local/enr" / f"{p['id']}.json").read_text())["chunks"]
        base += [Chunk(text=c["text"], doc_id=p["id"], chunk_id=f"{p['id']}:{i}",
                       embed_text=f"{c.get('section', '')}\n\n{c['text']}")
                 for i, c in enumerate(s)]
        enr += [Chunk(text=c["text"], doc_id=p["id"], chunk_id=f"{p['id']}:e{i}",
                      embed_text=render_meta(c)) for i, c in enumerate(e)]
        qa += [{"question": q["question"], "gold": q["evidence"], "doc_id": p["id"]}
               for q in p["questions"] if q["evidence"]]
    r = diagnose(base, qa, check=(), compare=[("bm25@enriched", "bm25")], enriched=enr,
                 bm25="study", tokenizer="cl100k_base", n_boot=10_000, seed=0, search_k=10_000)
    want = next(x for x in json.loads((RESULTS / "stats.json").read_text())["exploratory"]
                if (x["dataset"], x["retriever"], x["metric"], x["a"], x["b"])
                == ("qasper", "bm25", "rec@512t", "E", "S+T"))
    got = r.findings[0].comparison
    assert (r.settings["n_questions"], got.n_clusters) == (951, want["n_clusters"])
    assert (got.mean_a, got.mean_b) == pytest.approx((want["mean_a"], want["mean_b"]), abs=1e-12)
    assert got.diff == pytest.approx(want["diff"], abs=1e-12)
    assert got.ci95_bca == pytest.approx(want["ci95_bca"], abs=1e-9)
    assert got.p_perm == pytest.approx(want["p_perm"], abs=1e-12)
    assert r.findings[0].verdict == "better"
