"""Reference retrievers vs the v4 study harness.

- BM25 ``method="study"`` vs rank-bm25 ``BM25Okapi`` (what the study used): bit-identical
  scores. Runs whenever rank-bm25 is installed; the Qasper check also needs MDK_STUDY_DATA.
- Qwen3 reranker preset vs the study's cached scores on 50 Qasper pairs: slow (loads the
  model) and needs MDK_STUDY_DATA, e.g.

    MDK_STUDY_DATA=/path/to/data/v2 uv run --with pytest --with numpy --with rank-bm25 \
        --with sentence-transformers pytest tests/test_retrieval_parity.py --run-slow
"""
import json
import os
import random
from itertools import combinations
from pathlib import Path

import pytest

from mdkeychunker.models import Chunk

np = pytest.importorskip("numpy")

DATA = os.environ.get("MDK_STUDY_DATA", "")
needs_data = pytest.mark.skipif(not DATA, reason="set MDK_STUDY_DATA to run study parity tests")
QWEN3_KEY = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls|tmpl=v1|maxlen=2048"


def render_tc(c: dict) -> str:  # the study's "free prefix" index text (evaluate.render "tc")
    return f"{c.get('section', '')}\n\n{c['text']}"


def qasper_struct():
    data = Path(DATA)
    for p in json.loads((data / "qasper_v2.json").read_text(encoding="utf-8")):
        path = data / "cache" / "base" / "struct" / f"{p['id']}.json"
        if path.exists():
            yield p, json.loads(path.read_text(encoding="utf-8"))["chunks"]


def test_study_bm25_is_bit_identical_to_rank_bm25_on_random_corpora():
    rank_bm25 = pytest.importorskip("rank_bm25")
    from mdkeychunker.retrieval.bm25 import _Okapi

    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(40)]
    for _ in range(100):
        corpus = [[rng.choice(vocab[: rng.randint(3, 40)]) for _ in range(rng.randint(1, 60))]
                  for _ in range(rng.randint(1, 30))]
        ref = rank_bm25.BM25Okapi(corpus)
        ours = _Okapi(corpus, 1.5, 0.75, 0.25)
        for _ in range(5):
            q = [rng.choice(vocab) for _ in range(rng.randint(0, 6))]
            assert np.array_equal(ref.get_scores(q), np.array(ours.scores(q)))


def test_pure_python_argsort_matches_numpy_stable_argsort():
    from mdkeychunker.retrieval._base import argsort_desc

    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(0, 40)
        vals = [rng.choice([0.0, -0.0, 1.0, 2.5, rng.random()]) for _ in range(n)]
        want = list(np.argsort(-np.asarray(vals), kind="stable"))
        assert [i for i, _ in argsort_desc(vals)] == want
        assert [i for i, _ in argsort_desc(np.asarray(vals))] == want


@needs_data
def test_study_bm25_reproduces_harness_rankings_on_qasper():
    rank_bm25 = pytest.importorskip("rank_bm25")
    from mdkeychunker.retrieval import BM25Retriever
    from mdkeychunker.retrieval.bm25 import study_tokenize as words

    n = 0
    for p, raw in qasper_struct():
        docs = [render_tc(c) for c in raw]
        ref = rank_bm25.BM25Okapi([words(d) or ["_"] for d in docs])
        ours = BM25Retriever([Chunk(text=c["text"], embed_text=d) for c, d in zip(raw, docs)],
                             method="study")
        for q in p["questions"]:
            want = ref.get_scores(words(q["question"]))
            assert np.array_equal(want, np.array(ours.scores(q["question"]))), q["qid"]
            ranked = [i for i, _ in ours.rank(q["question"])]
            assert ranked == list(np.argsort(-want, kind="stable"))
            n += 1
    assert n >= 900


@needs_data
@pytest.mark.slow
def test_qwen3_reranker_preset_matches_study_scores():
    pytest.importorskip("sentence_transformers")
    from mdkeychunker.retrieval import Reranker
    from mdkeychunker.retrieval.rerank import RerankCache

    study = RerankCache(Path(DATA) / "rerank_cache", QWEN3_KEY)
    pairs = []
    for p, raw in qasper_struct():
        texts = [render_tc(c) for c in raw]
        for q in p["questions"]:
            cached = study.load(q["question"])
            pairs += [(q["question"], t, cached[h]) for t in texts
                      if (h := RerankCache.text_hash(t)) in cached]
        if len(pairs) >= 50:
            break
    pairs = pairs[:50]
    assert len(pairs) == 50, "not enough cached reranker scores"
    r = Reranker("qwen3-reranker-0.6b")
    assert r.model_key == QWEN3_KEY
    got = np.array([r.scores(q, [t])[0] for q, t, _ in pairs])
    want = np.array([s for *_, s in pairs])
    assert np.max(np.abs(got - want)) < 5e-3
    agree = np.mean([np.sign(want[i] - want[j]) == np.sign(got[i] - got[j])
                     for i, j in combinations(range(len(want)), 2)])
    assert agree >= 0.99
