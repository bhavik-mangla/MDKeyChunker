"""Optional LLM enrichment (``pip install "mdkeychunker[llm]"`` or ``[anthropic]``).

``enrich`` makes one LLM call per chunk that returns all fields (title, summary, keywords,
questions, ...) as JSON, and re-renders ``embed_text`` as those fields, a blank line and the
text. This is the v4 study's enrichment arm: the prompt is the 0.3 ``ENRICH_PROMPT``
verbatim, and the default ``rolling_keys=False`` is the study's E-noRK arm (rolling keys
"(not provided)"; secondary finding: the rolling dictionary does not change retrieval).

What the study found (FINDINGS.md; 7B indexing LLM, Qasper + FreshStack, English):

- confirmatory: enrichment beats the free section-path prefix by +5.5 points on Qasper under
  the untuned BM25+nomic hybrid (FreshStack +2.7, not significant), and is equivalent to
  contextual retrieval at 5.8-8.7x fewer input tokens per call;
- secondary: under qwen3-embedding or the Qwen3 reranker, enrichment is equivalent to the free
  prefix on both datasets; under BM25 alone the Qasper gain is largest (+8.7);
- exploratory: upgrading the retriever over free chunks gained more (+3.4 to +6.3).

So enrich for keyword-led (BM25) retrieval, and check your retriever first (``mk.diagnose``).

The study indexed fields plus text with no token cap. Here ``token_count`` is recounted on
the new ``embed_text``; chunks are never re-split (boundaries and ids stay fixed). When an
enriched ``embed_text`` exceeds ``max_tokens``, one :class:`EnrichmentOverflowWarning`
reports how many; ``max_field_tokens`` trims the generated fields (never the chunk text).
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import re
import tempfile
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Union, cast, runtime_checkable

from .chunking import Tokenizer, _resolve_tokenizer, _Tok
from .enricher import (
    ENRICH_PROMPT,
    FATAL_LLM_ERRORS,
    MAX_KEY_LEN,
    Enricher,
    _as_entities,
    _as_str,
    _as_str_list,
)
from .models import Chunk

__all__ = ["enrich", "aenrich", "estimate_cost", "CostEstimate", "OpenAICompatible",
           "Anthropic", "LLM", "EnrichmentWarning", "EnrichmentOverflowWarning", "FIELDS"]

# Bump to invalidate every cache entry; the template's hash is part of the key as well.
PROMPT_VERSION = "enrich-0.3"
_TEMPLATE_SHA = hashlib.sha256(ENRICH_PROMPT.encode()).hexdigest()[:12]
PARSE_RETRIES = 3          # as in the study: seed 0, then seeds 1..3
FIELDS = ("title", "summary", "keywords", "questions")  # the fields the study indexed
_RENDERABLE = ("title", "summary", "keywords", "questions", "entities")
NOT_PROVIDED = "(not provided)"

# Anthropic's contextual-retrieval prompt, as run in the study (benchmarks/chunking_study/build.py)
CR_PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""  # noqa: E501

# Output tokens per call measured in the study (qwen2.5 7B, Qasper): enrichment 190, CR 60.
# With gpt-oss 120B: 476 and 118.
STUDY_OUT_TOKENS = {"enrich": 190, "contextual": 60}
_PREV_SUMMARY_PLACEHOLDER = " ".join(["summary"] * 45)  # ~45 tokens, a typical 1-2 sentences


class _AbortedError(Exception):
    """Raised in calls queued behind an error that aborted the run."""


class EnrichmentWarning(UserWarning):
    """Some chunks kept empty fields (unparseable output or a failed call)."""


class EnrichmentOverflowWarning(UserWarning):
    """Some enriched ``embed_text`` values exceed the ``max_tokens`` the chunks were built with."""


# --------------------------------------------------------------------------- #
# LLM clients                                                                 #
# --------------------------------------------------------------------------- #

@runtime_checkable
class LLM(Protocol):
    """What ``enrich`` needs: an async completion and an id for the cache key.

    ``cache_id`` must change whenever outputs would (model, endpoint, temperature).
    A plain function ``prompt -> str`` (sync or async) works too; its id is its qualified name.
    """

    cache_id: str

    async def acomplete(self, prompt: str, *, json_mode: bool, max_tokens: int,
                        seed: int | None) -> str: ...


class _PerLoopClient:
    """SDK async clients hold connections bound to one event loop; make one per loop."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None

    def get(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._client, self._loop = self._factory(), loop
        return self._client


class OpenAICompatible:
    """OpenAI or any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenRouter, ...).

    Defaults follow the study: temperature 0, seed 0, JSON mode. Set ``temperature`` or
    ``seed`` to ``None`` for models that reject them; ``json_mode=False`` for servers
    without ``response_format``.

    >>> llm = OpenAICompatible("qwen2.5:7b", base_url="http://localhost:11434/v1")  # doctest: +SKIP
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, *,
                 temperature: float | None = 0.0, seed: int | None = 0, json_mode: bool = True,
                 timeout: float = 120.0, max_retries: int = 4,
                 extra_body: Mapping[str, Any] | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError('OpenAICompatible needs: pip install "mdkeychunker[llm]"') from e
        # Local servers need no key; the SDK reads OPENAI_API_KEY itself when it is set
        key = api_key or (None if "OPENAI_API_KEY" in os.environ else "not-needed")
        self.model, self.base_url = model, base_url
        self.temperature, self.seed, self.json_mode = temperature, seed, json_mode
        self.extra_body = dict(extra_body or {})
        self.cache_id = f"openai|{base_url or 'default'}|{model}|t={temperature}"
        self._clients = _PerLoopClient(lambda: AsyncOpenAI(
            api_key=key, base_url=base_url, timeout=timeout, max_retries=max_retries))

    async def acomplete(self, prompt: str, *, json_mode: bool, max_tokens: int,
                        seed: int | None) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens,
                                  "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed + (seed or 0)
        if json_mode and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = await self._clients.get().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class Anthropic:
    """Anthropic's Messages API (``pip install "mdkeychunker[anthropic]"``).

    The API has no seed and no JSON mode; the prompt asks for JSON only and the output is
    parsed leniently. ``api_key=None`` lets the SDK read ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, model: str, api_key: str | None = None, *,
                 temperature: float | None = 0.0, timeout: float = 120.0, max_retries: int = 4):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError('Anthropic needs: pip install "mdkeychunker[anthropic]"') from e
        self.model, self.temperature = model, temperature
        self.cache_id = f"anthropic|{model}|t={temperature}"
        self._clients = _PerLoopClient(lambda: AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries))

    async def acomplete(self, prompt: str, *, json_mode: bool, max_tokens: int,
                        seed: int | None) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens,
                                  "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        msg = await self._clients.get().messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")


class _FnLLM:
    def __init__(self, fn: Callable[[str], Union[str, Awaitable[str]]]):
        self.fn = fn
        self.cache_id = str(getattr(fn, "cache_id", "") or
                            f"fn|{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}")

    async def acomplete(self, prompt: str, *, json_mode: bool, max_tokens: int,
                        seed: int | None) -> str:
        if inspect.iscoroutinefunction(self.fn):
            return str(await self.fn(prompt))
        return str(await asyncio.to_thread(self.fn, prompt))


def _as_llm(llm: Any) -> LLM:
    if hasattr(llm, "acomplete"):
        if not getattr(llm, "cache_id", ""):
            llm.cache_id = type(llm).__qualname__
        return cast(LLM, llm)
    if callable(llm):
        return _FnLLM(llm)
    raise TypeError("llm must be OpenAICompatible, Anthropic, an object with acomplete(), "
                    "or a function prompt -> str")


# --------------------------------------------------------------------------- #
# Prompt, parsing, cache                                                      #
# --------------------------------------------------------------------------- #

def _prompt(c: Chunk, position: int, total: int, prev_summary: str, rolling_keys: str) -> str:
    return ENRICH_PROMPT.format(section_title=c.section_title or "(no section)",
                                position=position, total=total, prev_summary=prev_summary,
                                chunk_text=c.text, rolling_keys=rolling_keys)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _parse_json(text: str) -> dict[str, Any] | None:
    """The JSON object in an LLM reply: tolerant of code fences and surrounding prose."""
    s = _FENCE.sub("", text.strip())
    for candidate in (s, s[s.find("{"):s.rfind("}") + 1]):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class _Cache:
    """One JSON file per (prompt version, LLM, output cap, prompt); writes are atomic."""

    def __init__(self, directory: str | os.PathLike[str] | None):
        self.dir = Path(directory) if directory else None

    @staticmethod
    def key(llm_id: str, max_tokens: int, prompt: str) -> str:
        blob = json.dumps([PROMPT_VERSION, _TEMPLATE_SHA, llm_id, max_tokens, prompt])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.dir is not None
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.dir is None:
            return None
        try:
            value = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.dir is None:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, path)


def _apply(c: Chunk, result: dict[str, Any], known_keys: set[str]) -> None:
    """Field parsing exactly as the 0.3 LLMEnricher (the study's code path)."""
    c.title = _as_str(result.get("title"))
    c.summary = _as_str(result.get("summary"))
    c.keywords = _as_str_list(result.get("keywords"), split_commas=True)
    c.entities = _as_entities(result.get("entities"))
    c.questions = _as_str_list(result.get("questions"), split_commas=False)
    c.key = _as_str(result.get("key"), MAX_KEY_LEN).lower()
    c.related_keys = [k for k in (r.lower() for r in _as_str_list(result.get("related_keys"),
                                                                   split_commas=False))
                      if k in known_keys]


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

def _field_items(c: Chunk, name: str) -> list[str]:
    if name == "title":
        return [c.title] if c.title else []
    if name == "summary":
        return [c.summary] if c.summary else []
    if name == "entities":
        return [e["name"] for e in c.entities if e.get("name")]
    return list(getattr(c, name))


_JOIN = {"title": " ", "summary": " ", "keywords": ", ", "questions": " ", "entities": ", "}


def _field_text(c: Chunk, name: str, cap: int | None, tok: _Tok) -> str:
    items, sep = _field_items(c, name), _JOIN[name]
    text = sep.join(items)
    if cap is None or tok.count(text) <= cap:
        return text
    if name in ("keywords", "questions", "entities"):  # drop whole items from the end
        while items and tok.count(sep.join(items)) > cap:
            items.pop()
        if items:
            return sep.join(items)
        text = _field_items(c, name)[0]
    words = text.split()  # a single string: keep the longest word prefix that fits
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tok.count(" ".join(words[:mid])) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _render(c: Chunk, fields: Sequence[str], prefix: bool,
            caps: Mapping[str, int | None], tok: _Tok) -> str:
    """The study's "meta" rendering: one line per non-empty field, a blank line, the text."""
    head = [_field_text(c, f, caps.get(f), tok) for f in fields]
    if prefix and c.section_path:
        head.insert(0, " > ".join(c.section_path))
    lines = [h for h in head if h]
    return "\n".join(lines) + "\n\n" + c.text if lines else c.text


# --------------------------------------------------------------------------- #
# enrich / aenrich                                                            #
# --------------------------------------------------------------------------- #

_Notes = list[tuple[type[Warning], str]]


async def aenrich(chunks: Iterable[Chunk], llm: Any, **kwargs: Any) -> list[Chunk]:
    """Async :func:`enrich`; same arguments, returns new chunks in the input order."""
    out, notes = await _run(chunks, llm, **kwargs)
    for category, message in notes:
        warnings.warn(message, category, stacklevel=2)
    return out


async def _run(chunks: Iterable[Chunk], llm: Any, *, fields: Sequence[str] = FIELDS,
               concurrency: int = 8, cache_dir: str | os.PathLike[str] | None = None,
               rolling_keys: bool = False, prev_summary: bool = True,
               max_output_tokens: int = 1000, prefix: bool = False,
               max_tokens: int | None = 512,
               max_field_tokens: int | Mapping[str, int] | None = None,
               tokenizer: Tokenizer = None,
               on_error: Literal["warn", "raise"] = "warn") -> tuple[list[Chunk], _Notes]:
    """The work behind enrich/aenrich; warnings are returned so the caller's frame emits them."""
    fields = tuple(fields)
    bad = [f for f in fields if f not in _RENDERABLE]
    if bad:
        raise ValueError(f"unknown fields {bad}; choose from {_RENDERABLE}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if on_error not in ("warn", "raise"):
        raise ValueError("on_error must be 'warn' or 'raise'")
    model, tok, cache = _as_llm(llm), _resolve_tokenizer(tokenizer), _Cache(cache_dir)
    out = [_copy(c) for c in chunks]

    docs: dict[str, list[int]] = {}
    for i, c in enumerate(out):
        docs.setdefault(c.doc_id, []).append(i)
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()  # set by the first error that aborts the run
    failed: list[int] = []

    async def call(prompt: str) -> dict[str, Any] | None:
        key = cache.key(model.cache_id, max_output_tokens, prompt)
        hit = cache.get(key)
        if hit is not None:
            return hit
        for attempt in range(PARSE_RETRIES + 1):
            try:
                async with sem:
                    if stop.is_set():  # queued behind a call that aborted the run
                        raise _AbortedError
                    text = await model.acomplete(prompt, json_mode=True,
                                                 max_tokens=max_output_tokens, seed=attempt)
            except _AbortedError:
                raise
            except Exception as e:
                # Fatal errors (bad key, unknown model) fail identically on every chunk
                if on_error == "raise" or type(e).__name__ in FATAL_LLM_ERRORS:
                    stop.set()
                    raise
                return None  # transport errors: the SDK has already retried
            parsed = _parse_json(text)
            if parsed is not None:
                cache.put(key, parsed)
                return parsed
        return None

    async def one(i: int, pos: int, total: int, prev: str, keys: str,
                  known: set[str]) -> None:
        result = await call(_prompt(out[i], pos + 1, total, prev, keys))
        if result is None:
            failed.append(i)
        else:
            _apply(out[i], result, known)

    async def one_doc(idx: list[int]) -> None:
        state = Enricher()  # rolling-key bookkeeping, as in the study
        for pos, i in enumerate(idx):
            if not prev_summary:
                prev = NOT_PROVIDED
            elif pos == 0:
                prev = "(first chunk)"
            else:
                prev = out[idx[pos - 1]].summary or "(unavailable)"
            keys = state._format_rolling_keys() if rolling_keys else NOT_PROVIDED
            await one(i, pos, len(idx), prev, keys, set(state.rolling_keys))
            state._update_rolling_keys(out[i].key, pos)

    jobs: list[Awaitable[None]] = []
    for idx in docs.values():
        if prev_summary or rolling_keys:  # prompts depend on earlier outputs: in order
            jobs.append(one_doc(idx))
        else:  # independent prompts; positions are still within the document
            jobs.extend(one(i, pos, len(idx), NOT_PROVIDED, NOT_PROVIDED, set())
                        for pos, i in enumerate(idx))
    tasks = [asyncio.ensure_future(j) for j in jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:  # re-raise the error that aborted the run, not a queued call's _AbortedError
        raise next((e for e in errors if not isinstance(e, _AbortedError)), errors[0])

    caps = _caps(max_field_tokens, fields)
    over = 0
    for c in out:
        c.embed_text = _render(c, fields, prefix, caps, tok)
        c.token_count = tok.count(c.embed_text)
        over += max_tokens is not None and c.token_count > max_tokens
    notes: _Notes = []
    if failed:
        ids = ", ".join(out[i].chunk_id or str(i) for i in sorted(failed)[:5])
        notes.append((EnrichmentWarning,
                      f"{len(failed)} of {len(out)} chunks kept empty fields (unparseable output "
                      f"after {PARSE_RETRIES + 1} attempts, or a failed call); first: {ids}"))
    if over:
        notes.append((EnrichmentOverflowWarning,
                      f"{over} of {len(out)} enriched chunks exceed max_tokens={max_tokens} in "
                      "embed_text; chunks are not re-split. Pass max_field_tokens to trim the "
                      "generated fields (the study indexed fields + text uncapped)"))
    return out, notes


def _caps(max_field_tokens: int | Mapping[str, int] | None,
          fields: Sequence[str]) -> dict[str, int | None]:
    if max_field_tokens is None:
        return {}
    if isinstance(max_field_tokens, int):
        return {f: max_field_tokens for f in fields}
    return {f: max_field_tokens.get(f) for f in fields}


def _copy(c: Chunk) -> Chunk:
    new = copy.copy(c)
    for name in ("content_types", "keywords", "questions", "related_keys", "section_path"):
        setattr(new, name, list(getattr(c, name)))
    new.entities = [dict(e) for e in c.entities]
    new.metadata = dict(c.metadata)
    return new


def enrich(chunks: Iterable[Chunk], llm: Any, *, fields: Sequence[str] = FIELDS,
           concurrency: int = 8, cache_dir: str | os.PathLike[str] | None = None,
           rolling_keys: bool = False, prev_summary: bool = True, max_output_tokens: int = 1000,
           prefix: bool = False, max_tokens: int | None = 512,
           max_field_tokens: int | Mapping[str, int] | None = None, tokenizer: Tokenizer = None,
           on_error: Literal["warn", "raise"] = "warn") -> list[Chunk]:
    """One LLM call per chunk returning all fields; returns enriched copies, in order.

    Args:
        chunks: from :func:`mk.chunk` (any ``Chunk`` works; ``section_title`` feeds the prompt).
        llm: :class:`OpenAICompatible`, :class:`Anthropic`, an :class:`LLM`, or a function
            ``prompt -> str`` (sync or async).
        fields: generated fields put into ``embed_text``, in order (the prompt always asks
            for all of them, as in the study); also ``"entities"``.
        concurrency: maximum calls in flight.
        cache_dir: on-disk cache (one JSON file per call, keyed by prompt version, LLM
            ``cache_id``, output cap and the full prompt); reruns are free.
        rolling_keys: show the keys seen so far in the document (the study's E arm; no
            retrieval gain detected, v4 study, secondary). Off: the E-noRK arm.
        prev_summary: include the previous chunk's generated summary in the prompt, as the
            study did (its findings, e.g. enrichment equivalent to CR and one call equivalent
            to one call per field, were measured this way); calls within a document then run
            in order and documents run in parallel. ``False`` parallelises within a document
            too; that prompt is untested by the study.
        max_output_tokens: output cap per call (the study: 1000 for the 7B LLM).
        prefix: also put the section path first in ``embed_text`` (the study's E arm did not).
        max_tokens: pass the ``max_tokens`` you used in :func:`mk.chunk` (the default matches
            its default, 512); enriched ``embed_text`` over it gives one
            :class:`EnrichmentOverflowWarning`. ``None`` disables the check. ``token_count``
            is always recounted.
        max_field_tokens: trim each generated field (or ``{field: cap}``) to this many tokens
            in ``embed_text``; list fields drop items from the end. The chunk's fields
            themselves keep the full LLM output.
        tokenizer: as for :func:`mk.chunk`.
        on_error: ``"warn"``: a chunk whose output is unparseable after 4 attempts (seeds
            0-3), or whose call fails after the SDK's retries, keeps empty fields and is
            reported in one :class:`EnrichmentWarning`; ``"raise"``: the first failed call
            raises. Authentication / permission / unknown-model errors always raise.

    Chunk boundaries and ids never change. Runs inside Jupyter too (a running event loop
    is detected and the work moves to a worker thread).

    >>> import mdkeychunker as mk
    >>> fake = lambda p: '{"title": "Install", "keywords": ["pip", "venv"]}'
    >>> c = mk.enrich(mk.chunk("# Guide\\n\\nRun pip install."), fake)[0]
    >>> print(c.embed_text)
    Install
    pip, venv
    <BLANKLINE>
    # Guide
    <BLANKLINE>
    Run pip install.
    """
    coro = _run(chunks, llm, fields=fields, concurrency=concurrency, cache_dir=cache_dir,
                rolling_keys=rolling_keys, prev_summary=prev_summary,
                max_output_tokens=max_output_tokens, prefix=prefix, max_tokens=max_tokens,
                max_field_tokens=max_field_tokens, tokenizer=tokenizer, on_error=on_error)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        out, notes = asyncio.run(coro)
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:  # e.g. Jupyter: a loop is running
            out, notes = pool.submit(asyncio.run, coro).result()
    for category, message in notes:
        warnings.warn(message, category, stacklevel=2)
    return out


# --------------------------------------------------------------------------- #
# estimate_cost                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class CostEstimate:
    """Tokens and (optionally) dollars for indexing ``calls`` chunks with one method."""
    method: str
    calls: int
    input_tokens: int          # all input tokens, cached ones included
    cached_input_tokens: int   # part of input_tokens read from a prompt cache
    output_tokens: int
    usd: float | None

    @property
    def input_per_call(self) -> float:
        return self.input_tokens / self.calls if self.calls else 0.0

    def __str__(self) -> str:
        s = (f"{self.method}: {self.calls} calls, {self.input_tokens:,} input tokens "
             f"({self.cached_input_tokens:,} cached), {self.output_tokens:,} output tokens")
        return s + (f", ${self.usd:,.4f}" if self.usd is not None else "")


def estimate_cost(chunks: Iterable[Chunk], *,
                  method: Literal["enrich", "contextual"] = "enrich",
                  price_in: float | None = None, price_out: float | None = None,
                  price_cached_in: float | None = None, cached: bool = False,
                  out_tokens_per_call: int | None = None,
                  documents: Mapping[str, str] | None = None,
                  tokenizer: Tokenizer = None) -> CostEstimate:
    """Estimate the indexing cost of enrichment vs contextual retrieval (CR), before you pay.

    Input tokens are counted on the real prompts: :func:`enrich`'s (with a ~45-token
    previous summary) or the study's CR prompt (whole document + chunk, per call). Output
    tokens per call default to the study's measurements with qwen2.5 7B on Qasper
    (enrich 190, CR 60; with gpt-oss 120B: 476 and 118). Prices are dollars per million
    tokens. In the study CR sent 5.8x (Qasper papers) to 8.7x (FreshStack docs) the input
    tokens per call of enrichment (confirmatory finding 2: the two were equivalent for
    retrieval). The ratio depends on document length, because CR sends the whole document
    with every chunk: on short documents CR can be cheaper than enrichment (a few-line,
    one-chunk note: about 100 vs 660 input tokens with the default estimate, because the
    enrichment prompt's instructions dominate), so estimate on your own documents.

    ``cached=True`` models prompt caching for CR: after a document's first call its
    ``<document>`` prefix is billed at ``price_cached_in`` (default 10% of ``price_in``,
    roughly Anthropic's cache-read rate; OpenAI's discount differs). Cache-write surcharges
    and minimum cacheable lengths are not modelled. Enrichment prompts share no long
    prefix, so caching does not change them.

    ``documents`` maps ``doc_id`` to the full text for CR; without it the document is
    rebuilt from its chunks (exact for header chunks, an overestimate for overlapping
    windows). The default tokenizer is the chars/4 estimate; pass ``"tiktoken"`` for counts.

    >>> import mdkeychunker as mk
    >>> cs = mk.chunk("# A\\n\\n" + "word " * 2000, doc_id="d", headerless="structure")
    >>> e, cr = (mk.estimate_cost(cs, method=m) for m in ("enrich", "contextual"))
    >>> cr.input_tokens > 2 * e.input_tokens
    True
    """
    if method not in ("enrich", "contextual"):
        raise ValueError("method must be 'enrich' or 'contextual'")
    tok = _resolve_tokenizer(tokenizer)
    out_per_call = STUDY_OUT_TOKENS[method] if out_tokens_per_call is None else out_tokens_per_call
    docs: dict[str, list[Chunk]] = {}
    for c in chunks:
        docs.setdefault(c.doc_id, []).append(c)

    calls = in_tok = cached_tok = 0
    for doc_id, cs in docs.items():
        if method == "enrich":
            for pos, c in enumerate(cs):
                prev = "(first chunk)" if pos == 0 else (cs[pos - 1].summary
                                                         or _PREV_SUMMARY_PLACEHOLDER)
                in_tok += tok.count(_prompt(c, pos + 1, len(cs), prev, NOT_PROVIDED))
            calls += len(cs)
            continue
        doc = (documents or {}).get(doc_id)
        if doc is None:
            doc = "\n\n".join(c.text for c in sorted(cs, key=lambda c: c.start_char))
        doc_part = tok.count(f"<document>\n{doc}\n</document>\n")
        for c in cs:
            in_tok += tok.count(CR_PROMPT.format(doc=doc, chunk=c.text))
        if cached:
            cached_tok += doc_part * (len(cs) - 1)
        calls += len(cs)

    out_tok = calls * out_per_call
    usd = None
    if price_in is not None and price_out is not None:
        p_cached = price_in * 0.1 if price_cached_in is None else price_cached_in
        usd = ((in_tok - cached_tok) * price_in + cached_tok * p_cached
               + out_tok * price_out) / 1e6
    return CostEstimate(method, calls, in_tok, cached_tok, out_tok, usd)
