"""Every retrieval preset: a fast check of known version gates, and a slow load of each model.

The slow tests use whatever versions the resolver picks (the weekly CI job installs the
defaults), so a preset that breaks on a new transformers / sentence-transformers release
fails here instead of in a user's first diagnose() run.
"""
import math

import pytest

from mdkeychunker.models import Chunk
from mdkeychunker.retrieval import dense as dense_mod
from mdkeychunker.retrieval.dense import DENSE_PRESETS, _transformers_major, load_embedder
from mdkeychunker.retrieval.rerank import RERANK_PRESETS


def fake_versions(monkeypatch, transformers_version):
    real = dense_mod.version
    monkeypatch.setattr(dense_mod, "version", lambda name: transformers_version
                        if name == "transformers" else real(name))


def stop_before_loading(monkeypatch):
    def refuse(module, extra):
        raise RuntimeError(f"would load {module}")
    monkeypatch.setattr(dense_mod, "require", refuse)


@pytest.mark.parametrize("model", ["nomic", "nomic-ai/nomic-embed-text-v1.5"])
def test_nomic_on_transformers_5_fails_early_with_the_fix(monkeypatch, model):
    fake_versions(monkeypatch, "5.20.1")
    stop_before_loading(monkeypatch)
    with pytest.raises(ImportError, match=r"needs transformers<5 \(installed: 5\.20\.1\).*"
                                          r"pip install 'transformers<5'"):
        load_embedder(model)


def test_nomic_on_transformers_4_and_other_presets_pass_the_gate(monkeypatch):
    stop_before_loading(monkeypatch)
    fake_versions(monkeypatch, "4.57.6")
    with pytest.raises(RuntimeError, match="would load sentence_transformers"):
        load_embedder("nomic")
    fake_versions(monkeypatch, "5.20.1")
    for model in ("qwen3-embedding-0.6b", "mxbai", "some/other-model"):
        with pytest.raises(RuntimeError, match="would load sentence_transformers"):
            load_embedder(model)


def test_gate_is_silent_without_transformers(monkeypatch):
    def missing(name):
        raise dense_mod.PackageNotFoundError(name)
    monkeypatch.setattr(dense_mod, "version", missing)
    assert _transformers_major() is None
    stop_before_loading(monkeypatch)
    with pytest.raises(RuntimeError):
        load_embedder("nomic")  # falls through to the usual [dense] ImportError path


# --------------------------------------------------------------------------- slow: real models

DOCS = ["Paris is the capital and largest city of France.",
        "The mitochondria is the powerhouse of the cell.",
        "Bananas are rich in potassium."]
QUERY = "What is the capital of France?"


def norm(v):
    return math.sqrt(sum(float(x) ** 2 for x in v))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DENSE_PRESETS))
def test_every_dense_preset_loads_and_retrieves(name):
    pytest.importorskip("sentence_transformers")
    from mdkeychunker.retrieval import DenseRetriever

    preset = DENSE_PRESETS[name]
    major = _transformers_major()
    if preset.transformers_below and major is not None and major >= preset.transformers_below:
        with pytest.raises(ImportError, match=f"transformers<{preset.transformers_below}"):
            load_embedder(name, device="cpu")
        return
    embed = load_embedder(name, device="cpu")
    vecs = embed(["hello world", "another text"])
    # unit length up to bf16 precision: on transformers 5, Qwen3-Embedding loads in its
    # config's bfloat16 (norms ~1.003); DenseRetriever re-normalises in float32 anyway
    assert len(vecs) == 2 and all(norm(v) == pytest.approx(1.0, abs=1e-2) for v in vecs)
    r = DenseRetriever([Chunk(text=t, chunk_id=str(i)) for i, t in enumerate(DOCS)], name,
                       embed_fn=embed)
    assert r.search(QUERY, k=1)[0].chunk.chunk_id == "0"


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(RERANK_PRESETS))
def test_every_rerank_preset_loads_and_ranks(name):
    pytest.importorskip("sentence_transformers")
    from mdkeychunker.retrieval import Reranker

    scores = Reranker(name, device="cpu").scores(QUERY, DOCS)
    assert len(scores) == 3 and all(math.isfinite(s) for s in scores)
    assert scores[0] == max(scores)
