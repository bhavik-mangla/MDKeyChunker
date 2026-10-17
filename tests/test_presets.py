"""Every retrieval preset: a fast check of known version gates, and a slow load of each model.

The slow tests use whatever versions the resolver picks (the weekly CI job installs the
defaults), so a preset that breaks on a new transformers / sentence-transformers release
fails here instead of in a user's first diagnose() run.
"""
import json
import math
import os
from types import SimpleNamespace

import pytest

from mdkeychunker.models import Chunk
from mdkeychunker.retrieval import _base
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


# --------------------------------------------------------------------------- dtype plumbing

def fake_torch(monkeypatch):
    torch = SimpleNamespace(float32="F32", bfloat16="BF16")
    monkeypatch.setattr(_base, "require", lambda mod, extra: torch)


@pytest.mark.parametrize("tf,key", [("4.51.3", "torch_dtype"), ("4.55.4", "torch_dtype"),
                                    ("4.56.0", "dtype"), ("5.20.1", "dtype")])
def test_dtype_keyword_follows_the_transformers_version(monkeypatch, tf, key):
    import importlib.metadata as md

    fake_torch(monkeypatch)
    monkeypatch.setattr(md, "version", lambda name: tf)
    assert _base.model_dtype_kwargs("float32") == {key: "F32"}
    assert _base.model_dtype_kwargs("auto") == {key: "auto"}
    assert _base.model_dtype_kwargs(None) == {}


def test_kwargs_param_picks_the_supported_name():
    class Old:
        def __init__(self, name, automodel_args=None): ...

    class New:
        def __init__(self, name, model_kwargs=None): ...

    assert _base.kwargs_param(Old, "model_kwargs", "automodel_args") == "automodel_args"
    assert _base.kwargs_param(New, "model_kwargs", "automodel_args") == "model_kwargs"
    with pytest.raises(TypeError):
        _base.kwargs_param(Old, "model_kwargs")


def test_reranker_cache_key_keeps_the_study_key_for_float32():
    from mdkeychunker.retrieval import Reranker

    key = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls|tmpl=v1|maxlen=2048"
    assert Reranker("qwen3-reranker-0.6b").model_key == key
    assert Reranker("qwen3-reranker-0.6b", dtype=None).model_key != key
    assert Reranker("qwen3-reranker-0.6b", model_kwargs={"attn_implementation": "eager"}
                    ).model_key != key


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
    # float32 by default, so unit length holds tightly on every transformers version
    assert len(vecs) == 2 and all(norm(v) == pytest.approx(1.0, abs=1e-3) for v in vecs)
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


# --------------------------------------------------------------------------- slow: across versions

def preset_scores():
    """Scores of every preset that runs here on the toy query (float32 defaults)."""
    from mdkeychunker.retrieval import DenseRetriever, Reranker
    from mdkeychunker.retrieval.dense import _check_transformers, resolve_dense

    out = {}
    for name in sorted(DENSE_PRESETS):
        try:
            _check_transformers(name, resolve_dense(name))
        except ImportError:
            continue
        r = DenseRetriever([Chunk(text=t, chunk_id=str(i)) for i, t in enumerate(DOCS)], name,
                           device="cpu")
        out[name] = [h.score for h in sorted(r.search(QUERY, k=3), key=lambda h: h.chunk.chunk_id)]
    for name in sorted(RERANK_PRESETS):
        out[name] = Reranker(name, device="cpu").scores(QUERY, DOCS)
    return out


@pytest.mark.slow
def test_preset_scores_snapshot(tmp_path):
    """Write this environment's preset scores (CI's slow legs upload them for comparison)."""
    pytest.importorskip("sentence_transformers")
    from importlib.metadata import version

    scores = preset_scores()
    assert len(scores) >= len(RERANK_PRESETS) + 2
    out = os.environ.get("MDK_PRESET_SCORES_OUT") or str(tmp_path / "scores.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"transformers": version("transformers"), "scores": scores}, fh, indent=1)


def test_preset_scores_match_across_transformers_versions():
    """Float32 presets give the same scores on transformers 4 and 5 (CI's slow-compare job)."""
    spec = os.environ.get("MDK_PRESET_SCORES_COMPARE")
    if not spec:
        pytest.skip("set MDK_PRESET_SCORES_COMPARE=a.json,b.json (two slow-leg snapshots)")
    a, b = (json.load(open(p, encoding="utf-8")) for p in spec.split(","))
    assert a["transformers"].split(".")[0] != b["transformers"].split(".")[0], (
        "compare snapshots from different transformers major versions")
    common = sorted(set(a["scores"]) & set(b["scores"]))
    assert len(common) >= 4, common
    for name in common:
        assert a["scores"][name] == pytest.approx(b["scores"][name], abs=1e-6), (
            name, a["transformers"], b["transformers"])
