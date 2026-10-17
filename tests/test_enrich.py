"""Tests for mk.enrich / mk.aenrich / mk.estimate_cost with fake LLMs (no network)."""
import asyncio
import json
import sys
import types
import warnings

import pytest

import mdkeychunker as mk
from mdkeychunker.enrich import (
    CR_PROMPT,
    EnrichmentOverflowWarning,
    EnrichmentWarning,
    _parse_json,
)
from mdkeychunker.enricher import LLMEnricher

enrich_mod = sys.modules["mdkeychunker.enrich"]  # mk.enrich is the function

DOC = "\n\n".join(f"## Section {i}\n\n" + f"Body of section {i}. " * 30 for i in range(5))
RESULT = {"title": "T", "summary": "S.", "keywords": ["k1", "k2"], "questions": ["Q1?", "Q2?"],
          "entities": [{"name": "E", "type": "TECH"}], "key": "topic", "related_keys": []}


class Recorder:
    """Fake LLM: records prompts/seeds, answers from a function of (prompt, n-th call)."""

    def __init__(self, answer=None, cache_id="fake", delay=0.0):
        self.answer = answer or (lambda prompt, n: json.dumps(RESULT))
        self.cache_id, self.delay = cache_id, delay
        self.prompts, self.seeds = [], []
        self.in_flight = self.max_in_flight = 0

    async def acomplete(self, prompt, *, json_mode, max_tokens, seed):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            n = len(self.prompts)
            self.prompts.append(prompt)
            self.seeds.append(seed)
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.answer(prompt, n)
        finally:
            self.in_flight -= 1


def summary_per_chunk(prompt, n):
    return json.dumps({**RESULT, "summary": f"summary {n}", "key": f"key {n % 2}"})


# --------------------------------------------------------------------------- #
# Parity with the study's code path (0.3 LLMEnricher)                         #
# --------------------------------------------------------------------------- #

class _LegacyClient:
    def __init__(self):
        self.prompts, self.n = [], 0

    def call_json(self, prompt, max_tokens=1000):
        self.prompts.append(prompt)
        r = json.loads(summary_per_chunk(prompt, self.n))
        self.n += 1
        return r


class _NoRolling(LLMEnricher):  # the study's E-noRK ablation (build_v2.NoRollingKeysEnricher)
    def _format_rolling_keys(self):
        return "(not provided)"


@pytest.mark.parametrize("rolling", [False, True])
def test_prompts_match_the_study_enricher(rolling):
    chunks = mk.chunk(DOC, doc_id="d", max_tokens=None)
    legacy = _LegacyClient()
    (LLMEnricher if rolling else _NoRolling)(legacy).enrich_chunks(
        [enrich_mod._copy(c) for c in chunks])
    rec = Recorder(answer=summary_per_chunk)
    out = mk.enrich(chunks, rec, rolling_keys=rolling)
    assert rec.prompts == legacy.prompts
    assert "(previous" not in rec.prompts[1] and "summary 0" in rec.prompts[1]
    assert out[1].summary == "summary 1" and out[1].key == "key 1"


def test_embed_text_is_the_study_meta_rendering():
    out = mk.enrich(mk.chunk(DOC, doc_id="d"), Recorder())
    for c in out:  # benchmarks/chunking_study/evaluate.py render(mode="meta")
        head = [c.title, c.summary, ", ".join(c.keywords), " ".join(c.questions)]
        assert c.embed_text == "\n".join(h for h in head if h) + "\n\n" + c.text
        assert c.token_count == -(-len(c.embed_text) // 4)
    assert out[0].entities == [{"name": "E", "type": "TECH"}]


def test_fields_order_entities_and_prefix():
    c = mk.enrich(mk.chunk(DOC, doc_id="d"), Recorder(), fields=("keywords", "entities"),
                  prefix=True)[0]
    assert c.embed_text.split("\n")[:3] == ["Section 0", "k1, k2", "E"]
    with pytest.raises(ValueError):
        mk.enrich([], Recorder(), fields=("key",))


def test_inputs_are_not_mutated_and_ids_stay():
    chunks = mk.chunk(DOC, doc_id="d")
    before = [(c.chunk_id, c.embed_text, c.title) for c in chunks]
    out = mk.enrich(chunks, Recorder())
    assert [(c.chunk_id, c.embed_text, c.title) for c in chunks] == before
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks]
    assert [c.start_char for c in out] == [c.start_char for c in chunks]


# --------------------------------------------------------------------------- #
# Failures, retries, fatal errors                                             #
# --------------------------------------------------------------------------- #

def test_malformed_json_is_retried_with_study_seeds():
    rec = Recorder(answer=lambda p, n: "not json" if n < 2 else json.dumps(RESULT))
    out = mk.enrich(mk.chunk("# A\n\nshort body"), rec)
    assert rec.seeds == [0, 1, 2] and out[0].title == "T"


def test_unparseable_after_retries_keeps_empty_fields_and_warns_once():
    rec = Recorder(answer=lambda p, n: "never json")
    with pytest.warns(EnrichmentWarning, match="2 of 2"):
        out = mk.enrich(mk.chunk(DOC, doc_id="d", max_tokens=None)[:2], rec)
    assert rec.seeds == [0, 1, 2, 3] * 2
    assert all(not c.title and c.embed_text == c.text for c in out)


def test_lenient_json_parsing():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    assert _parse_json("[1, 2]") is None and _parse_json("") is None


def test_fatal_error_raises_and_cancels_the_rest():
    class AuthenticationError(Exception):
        pass

    def answer(p, n):
        raise AuthenticationError("bad key")

    rec = Recorder(answer=answer)
    with pytest.raises(AuthenticationError):
        mk.enrich(mk.chunk(DOC, doc_id="d"), rec, prev_summary=False, concurrency=1)
    assert len(rec.prompts) == 1  # no retries, and the queued chunks were cancelled


def test_transport_error_warns_or_raises():
    def answer(p, n):
        raise ConnectionError("down")

    with pytest.warns(EnrichmentWarning):
        mk.enrich(mk.chunk("# A\n\nbody"), Recorder(answer=answer))
    with pytest.raises(ConnectionError):
        mk.enrich(mk.chunk("# A\n\nbody"), Recorder(answer=answer), on_error="raise")


# --------------------------------------------------------------------------- #
# Concurrency and ordering                                                    #
# --------------------------------------------------------------------------- #

def _docs(n_docs, n_chunks=4):
    return [c for d in range(n_docs)
            for c in mk.chunk(DOC, doc_id=f"doc{d}", max_tokens=None)[:n_chunks]]


def test_documents_run_in_parallel_chunks_in_order():
    chunks = _docs(4)
    rec = Recorder(answer=summary_per_chunk, delay=0.01)
    out = mk.enrich(chunks, rec, concurrency=3)
    assert 1 < rec.max_in_flight <= 3
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks]
    # each document starts once; every later prompt carries a generated previous summary
    assert sum("**Previous Chunk Summary:** (first chunk)" in p for p in rec.prompts) == 4
    assert sum("**Previous Chunk Summary:** summary " in p for p in rec.prompts) == 12


def test_prev_summary_off_runs_every_chunk_in_parallel():
    chunks = _docs(1, n_chunks=5)
    rec = Recorder(delay=0.01)
    mk.enrich(chunks, rec, prev_summary=False, concurrency=8)
    assert rec.max_in_flight == 5
    assert all("**Previous Chunk Summary:** (not provided)" in p for p in rec.prompts)
    positions = sorted(p.split("**Chunk Position:** ")[1].split("\n")[0] for p in rec.prompts)
    assert positions == [f"{i} of 5 chunks" for i in range(1, 6)]


def test_concurrency_one_is_sequential():
    rec = Recorder(delay=0.005)
    mk.enrich(_docs(3), rec, concurrency=1)
    assert rec.max_in_flight == 1
    with pytest.raises(ValueError):
        mk.enrich([], rec, concurrency=0)


def test_sync_enrich_works_inside_a_running_loop():
    async def notebook_cell():
        return mk.enrich(mk.chunk("# A\n\nbody"), Recorder())

    assert asyncio.run(notebook_cell())[0].title == "T"


def test_aenrich_and_plain_functions():
    async def afn(prompt):
        return json.dumps(RESULT)

    out = asyncio.run(mk.aenrich(mk.chunk("# A\n\nbody"), afn))
    assert out[0].keywords == ["k1", "k2"]
    assert mk.enrich(mk.chunk("# A\n\nbody"), lambda p: json.dumps(RESULT))[0].title == "T"
    with pytest.raises(TypeError):
        mk.enrich(mk.chunk("# A\n\nbody"), 42)


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #

def test_cache_hit_miss_and_invalidation(tmp_path, monkeypatch):
    chunks = mk.chunk(DOC, doc_id="d")
    first = Recorder(answer=summary_per_chunk)
    a = mk.enrich(chunks, first, cache_dir=tmp_path)
    again = Recorder(answer=lambda p, n: pytest.fail("cache miss"))
    b = mk.enrich(chunks, again, cache_dir=tmp_path)
    assert [c.summary for c in a] == [c.summary for c in b] and again.prompts == []
    other = Recorder(cache_id="another-model")
    mk.enrich(chunks, other, cache_dir=tmp_path)
    assert len(other.prompts) == len(chunks)
    monkeypatch.setattr(enrich_mod, "PROMPT_VERSION", "enrich-next")
    bumped = Recorder()
    mk.enrich(chunks, bumped, cache_dir=tmp_path)
    assert len(bumped.prompts) == len(chunks)


def test_failures_are_not_cached_and_corrupt_entries_are_ignored(tmp_path):
    chunks = mk.chunk("# A\n\nbody")
    with pytest.warns(EnrichmentWarning):
        mk.enrich(chunks, Recorder(answer=lambda p, n: "x"), cache_dir=tmp_path)
    assert not list(tmp_path.rglob("*.json"))
    mk.enrich(chunks, Recorder(), cache_dir=tmp_path)
    for f in tmp_path.rglob("*.json"):
        f.write_text("{corrupt", encoding="utf-8")
    assert mk.enrich(chunks, Recorder(), cache_dir=tmp_path)[0].title == "T"


# --------------------------------------------------------------------------- #
# Overflow and field caps                                                     #
# --------------------------------------------------------------------------- #

LONG = {**RESULT, "summary": "word " * 300, "keywords": [f"keyword{i}" for i in range(80)]}


def test_warnings_point_at_the_caller():
    with pytest.warns(EnrichmentWarning) as rec:
        mk.enrich(mk.chunk("# A\n\nbody"), Recorder(answer=lambda p, n: "x"))
    assert rec[0].filename == __file__


def test_overflow_warns_once_with_count_and_never_resplits():
    chunks = mk.chunk(DOC, doc_id="d", max_tokens=128)
    with pytest.warns(EnrichmentOverflowWarning, match=f"{len(chunks)} of {len(chunks)}"):
        out = mk.enrich(chunks, Recorder(answer=lambda p, n: json.dumps(LONG)), max_tokens=128)
    assert len(out) == len(chunks) and all(c.token_count > 128 for c in out)


def test_max_field_tokens_trims_fields_not_text():
    chunks = mk.chunk(DOC, doc_id="d", max_tokens=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EnrichmentOverflowWarning)
        out = mk.enrich(chunks, Recorder(answer=lambda p, n: json.dumps(LONG)),
                        max_tokens=200, max_field_tokens=10)
    for c, orig in zip(out, chunks):
        assert c.embed_text.endswith("\n\n" + orig.text)
        assert len(c.keywords) == 80  # the chunk keeps the full LLM output
        head = c.embed_text[: -len(orig.text)]
        assert all(-(-len(line) // 4) <= 10 for line in head.split("\n") if line)
    per_field = mk.enrich(chunks[:1], Recorder(answer=lambda p, n: json.dumps(LONG)),
                          max_field_tokens={"keywords": 5}, max_tokens=None)[0]
    assert "word word" in per_field.embed_text and "keyword0, keyword1" in per_field.embed_text


# --------------------------------------------------------------------------- #
# Clients (fake SDK modules)                                                  #
# --------------------------------------------------------------------------- #

def _fake_openai(calls, clients):
    class Completions:
        async def create(self, **kw):
            calls.append(kw)
            msg = types.SimpleNamespace(content=json.dumps(RESULT))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class AsyncOpenAI:
        def __init__(self, **kw):
            clients.append(kw)
            self.chat = types.SimpleNamespace(completions=Completions())

    return types.SimpleNamespace(AsyncOpenAI=AsyncOpenAI)


def test_openai_compatible_request_and_per_loop_client(monkeypatch):
    calls, clients = [], []
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(calls, clients))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = mk.OpenAICompatible("qwen2.5:7b", base_url="http://localhost:11434/v1")
    mk.enrich(mk.chunk("# A\n\nbody"), llm)
    mk.enrich(mk.chunk("# B\n\nbody"), llm)  # a second asyncio.run: a fresh client
    assert len(clients) == 2 and clients[0]["api_key"] == "not-needed"
    kw = calls[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert (kw["temperature"], kw["seed"], kw["max_tokens"]) == (0.0, 0, 1000)
    assert "qwen2.5:7b" in llm.cache_id and "11434" in llm.cache_id
    bare = mk.OpenAICompatible("m", temperature=None, seed=None, json_mode=False)
    mk.enrich(mk.chunk("# A\n\nbody"), bare)
    assert not {"temperature", "seed", "response_format"} & set(calls[-1])


def test_anthropic_request(monkeypatch):
    calls = []

    class Messages:
        async def create(self, **kw):
            calls.append(kw)
            block = types.SimpleNamespace(type="text", text=json.dumps(RESULT))
            return types.SimpleNamespace(content=[block])

    class AsyncAnthropic:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(
        AsyncAnthropic=AsyncAnthropic))
    out = mk.enrich(mk.chunk("# A\n\nbody"), mk.Anthropic("claude-haiku-5-5"))
    assert out[0].title == "T" and calls[0]["temperature"] == 0.0
    assert "seed" not in calls[0] and "response_format" not in calls[0]


@pytest.mark.parametrize("cls, mod, extra", [("OpenAICompatible", "openai", "llm"),
                                             ("Anthropic", "anthropic", "anthropic")])
def test_clients_name_their_extra_when_sdk_missing(monkeypatch, cls, mod, extra):
    monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(ImportError, match=rf"mdkeychunker\[{extra}\]"):
        getattr(mk, cls)("m")


def test_mk_enrich_is_the_function_even_after_importing_the_submodule():
    import mdkeychunker.enrich  # noqa: F401
    assert callable(mk.enrich) and not isinstance(mk.enrich, types.ModuleType)


# --------------------------------------------------------------------------- #
# estimate_cost                                                               #
# --------------------------------------------------------------------------- #

def test_estimate_cost_counts_real_prompts():
    chunks = mk.chunk(DOC, doc_id="d", max_tokens=None)
    e = mk.estimate_cost(chunks, method="enrich")
    assert e.calls == len(chunks) and e.output_tokens == 190 * len(chunks)
    assert e.input_tokens > sum(c.token_count for c in chunks)
    cr = mk.estimate_cost(chunks, method="contextual", documents={"d": DOC})
    expected = sum(-(-len(CR_PROMPT.format(doc=DOC, chunk=c.text)) // 4) for c in chunks)
    assert cr.input_tokens == expected and cr.output_tokens == 60 * len(chunks)


def test_cr_to_enrich_ratio_grows_with_document_length():
    ratios = []
    for n in (5, 40):
        md = "\n\n".join(f"## S{i}\n\n" + "text " * 250 for i in range(n))
        cs = mk.chunk(md, doc_id="d")
        e, cr = (mk.estimate_cost(cs, method=m) for m in ("enrich", "contextual"))
        ratios.append(cr.input_per_call / e.input_per_call)
    assert 1 < ratios[0] < ratios[1]


def test_estimate_cost_prices_and_prompt_caching():
    cs = mk.chunk(DOC, doc_id="d", max_tokens=None)
    plain = mk.estimate_cost(cs, method="contextual", price_in=1.0, price_out=2.0)
    cached = mk.estimate_cost(cs, method="contextual", price_in=1.0, price_out=2.0, cached=True)
    assert plain.usd == pytest.approx((plain.input_tokens + 2 * plain.output_tokens) / 1e6)
    assert cached.cached_input_tokens > 0 and cached.usd < plain.usd
    assert cached.input_tokens == plain.input_tokens
    e_cached = mk.estimate_cost(cs, method="enrich", cached=True)
    assert e_cached.cached_input_tokens == 0 and e_cached.usd is None
    assert "contextual:" in str(cached) and "$" in str(cached)
    assert mk.estimate_cost(cs, out_tokens_per_call=476).output_tokens == 476 * len(cs)
    with pytest.raises(ValueError):
        mk.estimate_cost(cs, method="both")
