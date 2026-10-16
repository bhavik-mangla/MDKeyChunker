"""0.4 packaging: zero-dependency core, no import or runtime side effects, deprecations."""
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import mdkeychunker as mk
from mdkeychunker import pipeline as pipeline_mod
from mdkeychunker.config import Config
from mdkeychunker.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ["torch", "openai", "anthropic", "sentence_transformers", "transformers", "tiktoken",
         "dotenv", "bm25s", "Stemmer", "spacy", "numpy", "langchain_core", "llama_index"]


def _fresh_python(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return out.stdout


def test_import_loads_no_optional_dependency():
    code = ("import sys, json, mdkeychunker as mk; mk.chunk('# A\\n\\nb'); "
            f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))")
    assert json.loads(_fresh_python(code)) == []


def test_import_configures_no_logging_and_reads_no_env(tmp_path):
    (tmp_path / ".env").write_text("LLM_MODEL=from-dotenv\n")
    code = ("import logging, os, mdkeychunker; "
            "print(len(logging.getLogger().handlers), os.environ.get('LLM_MODEL'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, check=True, env={"PYTHONPATH": str(ROOT)})
    assert out.stdout.split() == ["0", "None"]


def test_public_names_resolve_lazily():
    for name in mk.__all__:
        assert getattr(mk, name) is not None, name
    assert {"chunk", "chunk_file", "window", "Chunk", "Hit", "GoldSpan", "Pipeline",
            "Config"} <= set(mk.__all__)
    assert set(dir(mk)) >= set(mk.__all__)
    with pytest.raises(AttributeError):
        mk.not_a_name  # noqa: B018
    assert isinstance(mk.__version__, str)


def test_planned_names_map_to_modules():
    # Names whose modules have not landed yet must not appear in __all__ (import * works)
    for name, mod in mk._LAZY.items():
        present = (ROOT / "mdkeychunker" / f"{mod}.py").exists() or \
            (ROOT / "mdkeychunker" / mod / "__init__.py").exists()
        assert (name in mk.__all__) == present, name


def test_py_typed_marker_is_shipped():
    assert (ROOT / "mdkeychunker" / "py.typed").exists()
    assert 'mdkeychunker = ["py.typed"]' in (ROOT / "pyproject.toml").read_text()


def test_core_has_no_dependencies():
    text = (ROOT / "pyproject.toml").read_text()
    assert "\ndependencies = []\n" in text


# --------------------------------------------------------------------------- #
# Legacy Pipeline / Config                                                    #
# --------------------------------------------------------------------------- #

def test_pipeline_is_deprecated_and_adds_no_log_handlers(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "LLMClient", lambda config: object())
    before = list(logging.getLogger().handlers)
    with pytest.warns(DeprecationWarning, match="mk.chunk"):
        Pipeline(Config(merge_by_keys=False))
    assert logging.getLogger().handlers == before


def test_pipeline_spacy_mode_never_builds_an_llm_client(monkeypatch):
    def boom(config):
        raise AssertionError("LLMClient built in spaCy mode")

    class FakeSpacy(pipeline_mod.Enricher):
        def enrich_chunks(self, chunks):
            return chunks

    monkeypatch.setattr(pipeline_mod, "LLMClient", boom)
    monkeypatch.setattr(pipeline_mod, "SpacyEnricher", FakeSpacy)
    with pytest.warns(DeprecationWarning):
        p = Pipeline(Config(), enricher_mode="spacy")
    assert p.llm is None


def test_merge_by_keys_defaults_off_and_warns_once_in_llm_mode(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "LLMClient", lambda config: object())
    assert not Config().merge_by_keys
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Pipeline(Config())
    assert sum("merge_by_keys now defaults to False" in str(x.message) for x in w) == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Pipeline(Config(merge_by_keys=True))
    assert not any("merge_by_keys" in str(x.message) for x in w)


def test_min_orphan_size_is_deprecated():
    with pytest.warns(DeprecationWarning, match="min_orphan_size"):
        Config(min_orphan_size=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config()


def test_from_env_no_longer_reads_dotenv_implicitly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    (tmp_path / ".env").write_text("LLM_MODEL=from-dotenv\n")
    with pytest.warns(DeprecationWarning, match="env_file"):
        cfg = Config.from_env()
    assert cfg.llm_model == "gpt-4o-mini"


def test_from_env_reads_an_explicit_env_file(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("MERGE_BY_KEYS", raising=False)
    f = tmp_path / "custom.env"
    f.write_text("LLM_MODEL=from-file\nMERGE_BY_KEYS=true\n")
    cfg = Config.from_env(str(f))
    monkeypatch.delenv("LLM_MODEL", raising=False)  # load_dotenv set it in os.environ
    monkeypatch.delenv("MERGE_BY_KEYS", raising=False)
    assert cfg.llm_model == "from-file" and cfg.merge_by_keys is True


def test_from_env_merge_unset_means_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERGE_BY_KEYS", raising=False)
    assert Config.from_env().merge_by_keys is None
    monkeypatch.setenv("MERGE_BY_KEYS", "false")
    assert Config.from_env().merge_by_keys is False


def test_llm_client_without_openai_names_the_extra(monkeypatch):
    from mdkeychunker.llm_client import LLMClient
    monkeypatch.setitem(sys.modules, "openai", None)  # makes `import openai` fail
    with pytest.raises(ImportError, match=r"mdkeychunker\[llm\]"):
        LLMClient(Config(llm_provider="openai"))
