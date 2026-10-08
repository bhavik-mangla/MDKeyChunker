"""v2 chunk-set builder (ANALYSIS_PLAN_v2 sec. 2), routed through llm_api.

Caches: data/v2/cache/<model_tag>/<variant>/<doc_id>.json, so local-7B and
strong-model outputs never mix. Chunk sets that need no LLM live under the
model tag "base".

Variants
  fixed512   512-character windows, no overlap                       (F512c)
  fixedtok   256-token windows, 32-token overlap                     (F256t)
  struct     MDKeyChunker Stage 1                                    (S; S+T at index time)
  parapack   (--parapack) blank-line paragraphs packed greedily up to the Stage-1
             max size, headers treated as ordinary paragraphs; fenced code is
             never split; an oversize paragraph stays whole
  fixedlen   (--fixedlen) token windows, no overlap, window = median Stage-1
             chunk length (cl100k tokens) over the dataset's selected docs
  enr        struct + single-call enrichment (mdkeychunker LLMEnricher, rolling
             keys) through an adapter around llm_api.complete           (E / E-strong)
  cr         struct + contextual-retrieval prefix (build.CR_PROMPT)    (CR / CR-strong)

Each document is built by one worker from start to end, so the CR calls of a
document run back to back and its prompt prefix stays in the KV/prefix cache.

LLM failures (plan sec. 7): unparseable/empty output is retried up to 3 times
(seeds 1..3 after seed 0), then the chunk keeps empty fields, is flagged and
counted. Transport/HTTP failures after llm_api's own retries fail the document
(not cached, a rerun resumes). BudgetExceeded stops the whole run (exit 3).

The run refuses to start unless results/v2/frozen_config.json is absent (it is
then written) or identical to the current configuration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import tiktoken
from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config
from mdkeychunker.enricher import ENRICH_PROMPT, LLMEnricher

import answer
import llm_api
from build import CR_PROMPT, chunk_dict, fixed_chars, fixed_tokens

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
CACHE_V2 = V2 / "cache"
FROZEN = ROOT / "results" / "v2" / "frozen_config.json"
ENC = tiktoken.get_encoding("cl100k_base")

OLLAMA_URL = "http://127.0.0.1:11435"
NUM_CTX = {"qasper": 16384, "freshstack": 32768}
BASE_TAG = "base"
BASE_VARIANTS = ("fixed512", "fixedtok", "struct", "parapack", "fixedlen")
LLM_VARIANTS = ("enr", "cr")
PARSE_RETRIES = 3
FIXED512, FIXEDTOK_SIZE, FIXEDTOK_OVERLAP = 512, 256, 32


@dataclass(frozen=True)
class ModelConfig:
    tag: str
    spec: str
    reasoning: Any
    temperature: float = 0.0
    seed: int = 0
    enrich_max_tokens: int = 1000
    cr_max_tokens: int = 120

    def options(self) -> dict:
        s = llm_api.ModelSpec.parse(self.spec)
        o = {"backend": s.backend, "model": s.model, "providers": list(s.providers),
             "temperature": self.temperature, "seed": self.seed, "reasoning": self.reasoning,
             "enrich_max_tokens": self.enrich_max_tokens, "cr_max_tokens": self.cr_max_tokens}
        if s.backend == "ollama":
            o.update(url=OLLAMA_URL, think=False, num_ctx=NUM_CTX)
        else:
            o.update(allow_fallbacks=False, data_collection="deny", banned=llm_api.BANNED_PROVIDERS)
        return o


MODELS = {
    "local": ModelConfig("local", "ollama:qwen2.5:7b", reasoning=None),
    # Reasoning tokens count against max_tokens on OpenRouter, so the strong arm
    # gets more room; the prompt still asks for a short answer.
    "strong": ModelConfig("strong", "openrouter:openai/gpt-oss-120b@deepinfra/bf16", reasoning="low",
                          enrich_max_tokens=2000, cr_max_tokens=600),
}


class BudgetStop(RuntimeError):
    """Raised inside workers after another worker hit the budget cap."""


def configure_llm_api(dataset: str) -> None:
    """One Ollama URL and one num_ctx per run (Ollama reloads the model when num_ctx changes)."""
    llm_api.OLLAMA_URL = OLLAMA_URL
    llm_api.NUM_CTX = NUM_CTX[dataset]


# --------------------------------------------------------------------------- frozen config

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_state(path: Path) -> dict:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def ollama_digest(model: str, url: str = OLLAMA_URL) -> str | None:
    """Digest (weights + quantization) of the local model; a metadata GET, not a generation."""
    import requests
    try:
        tags = requests.get(f"{url}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        return None
    return next((m.get("digest") for m in tags if m.get("name") == model), None)


def current_config(digest_fn: Callable[[str], str | None] = ollama_digest) -> dict:
    import mdkeychunker
    pkg_root = Path(mdkeychunker.__file__).resolve().parent.parent
    models = {}
    for tag, m in MODELS.items():
        o = m.options()
        if o["backend"] == "ollama":
            o["digest"] = digest_fn(o["model"])
        models[tag] = {"spec": m.spec, **o}
    cfg = Config()
    return {
        "plan": "ANALYSIS_PLAN_v2",
        "models": models,
        "generator": {"spec": MODELS["local"].spec, "budget_tokens": {"qasper": 512, "freshstack": 1024},
                      "settings": answer.GEN_SETTINGS},
        "prompts_sha256": {"enrich": sha256(ENRICH_PROMPT), "cr": sha256(CR_PROMPT),
                           **{f"answer_{k}": v for k, v in answer.prompt_hashes().items()}},
        "chunking": {"fixed512": FIXED512, "fixedtok": [FIXEDTOK_SIZE, FIXEDTOK_OVERLAP],
                     "struct": {"min_chunk_size": cfg.min_chunk_size, "max_chunk_size": cfg.max_chunk_size},
                     "parapack_max_chars": cfg.max_chunk_size, "fixedlen": "median struct chunk tokens per dataset",
                     "tokenizer": "cl100k_base"},
        "retry": {"parse_retries": PARSE_RETRIES, "retry_seeds": list(range(1, PARSE_RETRIES + 1))},
        "selection_seed": 2027,
        "git": {"harness": git_state(ROOT), "mdkeychunker": git_state(pkg_root)},
        "mdkeychunker_version": getattr(mdkeychunker, "__version__", None),
    }


def _comparable(cfg: dict, ignore_git: bool) -> dict:
    c = copy.deepcopy(cfg)
    c.pop("created_at", None)
    if ignore_git:
        c.pop("git", None)
    else:  # dirty flags are informational only
        for v in (c.get("git") or {}).values():
            v.pop("dirty", None)
    return c


class FrozenConfigMismatch(RuntimeError):
    pass


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
        return out
    return [] if a == b else [path]


def check_frozen(cfg: dict, path: Path = FROZEN, write: bool = True, ignore_git: bool = False) -> dict:
    """Write the frozen config if absent; otherwise refuse when anything differs."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        # An unreachable Ollama (e.g. during a strong-only run) cannot report the
        # digest; that is not evidence of a change, so keep the frozen value.
        for tag, m in (cfg.get("models") or {}).items():
            if "digest" in m and m["digest"] is None and (old.get("models") or {}).get(tag, {}).get("digest"):
                print(f"WARN {tag} model digest unavailable; frozen digest not re-verified", flush=True)
                m["digest"] = old["models"][tag]["digest"]
        diffs = _diff(_comparable(old, ignore_git), _comparable(cfg, ignore_git))
        if diffs:
            raise FrozenConfigMismatch(
                f"{path} differs from the current configuration at: {', '.join(diffs)}. "
                "Refusing to run. Revert the change, or log a deviation and move the frozen file aside.")
        return old
    if not write:
        raise FrozenConfigMismatch(f"{path} does not exist; run build_v2.py first to freeze the config")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**cfg, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    return cfg


# --------------------------------------------------------------------------- non-LLM chunkers

def paragraphs(md: str) -> list[str]:
    """Blank-line separated blocks; a fenced code block is never split."""
    out, cur, fence = [], [], None
    for line in md.replace("\r\n", "\n").split("\n"):
        s = line.lstrip()
        if fence is None and (s.startswith("```") or s.startswith("~~~")):
            fence = s[:3]
        elif fence is not None and s.startswith(fence):
            fence = None
            cur.append(line)
            continue
        if fence is None and not line.strip():
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def parapack(md: str, max_size: int | None = None) -> list[dict]:
    """Greedy paragraph packing up to max_size chars (Stage-1 max), ignoring headers."""
    max_size = max_size or Config().max_chunk_size
    chunks, cur = [], []
    for p in paragraphs(md):
        if cur and len("\n\n".join(cur + [p])) > max_size:
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return [{"text": t} for t in chunks]


def fixed_len(md: str, size: int) -> list[dict]:
    toks = ENC.encode(md)
    return [{"text": ENC.decode(toks[i:i + size])} for i in range(0, max(len(toks), 1), size)]


def median_struct_tokens(mds: list[str], cfg: Config | None = None) -> int:
    chunker = MarkdownChunker(cfg or Config(log_level="WARNING"))
    lens = [len(ENC.encode(c.text)) for md in mds for c in chunker.chunk(md)]
    return int(round(statistics.median(lens))) if lens else FIXEDTOK_SIZE


# --------------------------------------------------------------------------- LLM adapters

class LLMAdapter:
    """Drop-in for mdkeychunker's LLMClient.call_json, routed through llm_api.complete.

    LLMEnricher swallows exceptions from call_json, so a fatal error (budget,
    transport) is remembered and re-raised by the caller after enrich_chunks."""

    def __init__(self, model: ModelConfig, complete_fn: Callable | None = None,
                 stop: threading.Event | None = None, tag: str = "enrich"):
        self.model = model
        self.complete = complete_fn or llm_api.complete
        self.stop = stop or threading.Event()
        self.tag = tag
        self.fatal: BaseException | None = None
        self.calls = self.in_tok = self.out_tok = self.parse_retries = 0
        self.cost = 0.0
        self.seconds = 0.0
        self.providers: set = set()
        self.failed: list[int] = []      # 0-based indices of calls that kept empty fields
        self._n = 0

    def _call(self, prompt: str, json_mode: bool, max_tokens: int, seed: int) -> dict:
        if self.stop.is_set():
            self.fatal = self.fatal or BudgetStop("run stopped (budget)")
        if self.fatal:
            raise self.fatal
        t = time.time()
        try:
            r = self.complete(self.model.spec, prompt, json_mode=json_mode, max_tokens=max_tokens,
                              temperature=self.model.temperature, seed=seed,
                              reasoning=self.model.reasoning, tag=self.tag)
        except llm_api.BudgetExceeded as e:
            self.fatal = e
            self.stop.set()
            raise
        except Exception as e:
            self.fatal = e
            raise
        self.seconds += time.time() - t
        self.calls += 1
        self.in_tok += r.get("in_tokens") or 0
        self.out_tok += r.get("out_tokens") or 0
        self.cost += r.get("cost_usd") or 0.0
        if r.get("provider"):
            self.providers.add(r["provider"])
        return r

    def call_json(self, prompt: str, max_tokens: int = 1000):
        idx, self._n = self._n, self._n + 1
        for attempt in range(PARSE_RETRIES + 1):
            seed = self.model.seed if attempt == 0 else attempt
            r = self._call(prompt, True, self.model.enrich_max_tokens, seed)
            if isinstance(r.get("parsed_json"), dict):
                return r["parsed_json"]
            if attempt < PARSE_RETRIES:
                self.parse_retries += 1
        self.failed.append(idx)
        return None

    def call_text(self, prompt: str, max_tokens: int) -> str | None:
        idx, self._n = self._n, self._n + 1
        for attempt in range(PARSE_RETRIES + 1):
            seed = self.model.seed if attempt == 0 else attempt
            text = (self._call(prompt, False, max_tokens, seed).get("text") or "").strip()
            if text:
                return text
            if attempt < PARSE_RETRIES:
                self.parse_retries += 1
        self.failed.append(idx)
        return None

    def stats(self) -> dict:
        return {"spec": self.model.spec, "calls": self.calls, "seconds": round(self.seconds, 1),
                "in_tokens": self.in_tok, "out_tokens": self.out_tok, "cost_usd": round(self.cost, 6),
                "providers": sorted(self.providers), "parse_retries": self.parse_retries,
                "failed_chunks": len(self.failed)}


def enrich(struct_chunks: list, adapter: LLMAdapter) -> tuple[list[dict], dict]:
    enriched = LLMEnricher(adapter).enrich_chunks(copy.deepcopy(struct_chunks))  # type: ignore[arg-type]
    if adapter.fatal:
        raise adapter.fatal
    failed = set(adapter.failed)
    out = [{**chunk_dict(c), "llm_failed": i in failed} for i, c in enumerate(enriched)]
    return out, adapter.stats()


def contextual_prefixes(md: str, chunks: list[dict], adapter: LLMAdapter) -> tuple[list[dict], dict]:
    """Document first, chunks of one document back to back (prefix-cache reuse)."""
    out = []
    for c in chunks:
        ctx = adapter.call_text(CR_PROMPT.format(doc=md, chunk=c["text"]), adapter.model.cr_max_tokens)
        out.append({**c, "context": ctx or "", "llm_failed": ctx is None})
    return out, adapter.stats()


# --------------------------------------------------------------------------- documents

def load_docs(dataset: str, selection: dict, domains: list[str] | None = None) -> list[dict]:
    if dataset == "qasper":
        papers = {p["id"]: p for p in json.loads((V2 / "qasper_v2.json").read_text())}
        return [{"id": i, "markdown": papers[i]["markdown"], "domain": "qasper"}
                for i in selection["qasper"]["paper_ids"]]
    docs = {d["id"]: d for d in json.loads((V2 / "freshstack_v2_docs.json").read_text())}
    out = []
    for dom in domains or ("laravel", "angular", "yolo"):
        out += [{"id": i, "markdown": docs[i]["markdown"], "domain": dom}
                for i in selection["freshstack"][dom]["doc_ids"]]
    return out


def cache_path(tag: str, variant: str, doc_id: str, cache: Path = CACHE_V2) -> Path:
    return Path(cache) / tag / variant / f"{doc_id}.json"


def variant_tag(variant: str, model_tag: str) -> str:
    return model_tag if variant in LLM_VARIANTS else BASE_TAG


def missing(doc_ids: list[str], variants: list[str], model_tag: str, cache: Path = CACHE_V2) -> list[str]:
    return [i for i in doc_ids
            if not all(cache_path(variant_tag(v, model_tag), v, i, cache).exists() for v in variants)]


@dataclass
class Builder:
    dataset: str
    model_tag: str
    variants: list[str]
    cache: Path = CACHE_V2
    fixedlen_size: int | None = None
    complete_fn: Callable | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    log: Callable[[str], None] = lambda s: print(s, flush=True)

    @property
    def model(self) -> ModelConfig:
        return MODELS[self.model_tag]

    def _get(self, variant: str, doc_id: str):
        p = cache_path(variant_tag(variant, self.model_tag), variant, doc_id, self.cache)
        return json.loads(p.read_text()) if p.exists() else None

    def _save(self, variant: str, doc_id: str, chunks: list[dict], stats: dict | None = None) -> None:
        p = cache_path(variant_tag(variant, self.model_tag), variant, doc_id, self.cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunks": chunks, "stats": stats or {}}))
        tmp.replace(p)  # atomic: a crash never leaves a half-written cache file

    def build_doc(self, doc: dict) -> dict:
        if self.stop.is_set():
            raise BudgetStop("run stopped (budget)")
        did, md = doc["id"], doc["markdown"]
        v = set(self.variants)
        if "fixed512" in v and not self._get("fixed512", did):
            self._save("fixed512", did, fixed_chars(md, FIXED512))
        if "fixedtok" in v and not self._get("fixedtok", did):
            self._save("fixedtok", did, fixed_tokens(md, FIXEDTOK_SIZE, FIXEDTOK_OVERLAP))
        if "parapack" in v and not self._get("parapack", did):
            self._save("parapack", did, parapack(md), {"max_chars": Config().max_chunk_size})
        if "fixedlen" in v and not self._get("fixedlen", did):
            if not self.fixedlen_size:
                raise ValueError("fixedlen needs fixedlen_size")
            self._save("fixedlen", did, fixed_len(md, self.fixedlen_size), {"window": self.fixedlen_size})
        struct = MarkdownChunker(Config(log_level="WARNING")).chunk(md)
        if not self._get("struct", did):
            self._save("struct", did, [chunk_dict(c) for c in struct])
        info: dict = {}
        if "enr" in v and not self._get("enr", did):
            ad = LLMAdapter(self.model, self.complete_fn, self.stop, tag=f"enrich-{self.dataset}")
            chunks, st = enrich(struct, ad)
            self._save("enr", did, chunks, st)
            info["enr"] = st
        if "cr" in v and not self._get("cr", did):
            ad = LLMAdapter(self.model, self.complete_fn, self.stop, tag=f"cr-{self.dataset}")
            chunks, st = contextual_prefixes(md, self._get("struct", did)["chunks"], ad)
            self._save("cr", did, chunks, st)
            info["cr"] = st
        return info

    def run(self, docs: list[dict], workers: int = 4) -> dict:
        """Build every doc; returns {"done", "failed", "budget_stop", "parse_failures"}."""
        res = {"done": 0, "failed": [], "budget_stop": False, "parse_failures": 0, "cost_usd": 0.0}
        lock = threading.Lock()

        def one(doc: dict) -> None:
            if self.stop.is_set():
                return
            t = time.time()
            try:
                info = self.build_doc(doc)
            except (llm_api.BudgetExceeded, BudgetStop) as e:
                self.stop.set()
                with lock:
                    res["budget_stop"] = True
                self.log(f"{doc['id']} BUDGET_STOP: {e}")
                return
            except Exception as e:  # doc not cached; a rerun resumes it
                with lock:
                    res["failed"].append(doc["id"])
                self.log(f"{doc['id']} FAILED: {type(e).__name__}: {llm_api._redact(str(e))[:200]}")
                return
            with lock:
                res["done"] += 1
                for st in info.values():
                    res["parse_failures"] += st.get("failed_chunks", 0)
                    res["cost_usd"] += st.get("cost_usd", 0.0)
            extra = " ".join(f"{k}:{s['calls']}calls/{s['failed_chunks']}fail/${s['cost_usd']:.4f}"
                             for k, s in info.items())
            self.log(f"{doc['id']} done in {time.time() - t:.0f}s {extra}".rstrip())

        with ThreadPoolExecutor(max(1, workers)) as ex:
            list(ex.map(one, docs))
        return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["qasper", "freshstack"])
    ap.add_argument("--model-tag", default="local", choices=sorted(MODELS))
    ap.add_argument("--variants", default="fixed512,fixedtok,struct,enr,cr")
    ap.add_argument("--parapack", action="store_true", help="also build the parapack control")
    ap.add_argument("--fixedlen", action="store_true", help="also build the fixedlen control")
    ap.add_argument("--domains", default="laravel,angular,yolo", help="FreshStack domains")
    ap.add_argument("--sample", type=int, default=0,
                    help="random subset of N docs (seed 2027), e.g. the pre-declared CR-strong reduction")
    ap.add_argument("--limit", type=int, default=0, help="first N docs (smoke runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selection", default=str(V2 / "selection.json"))
    ap.add_argument("--frozen", default=str(FROZEN))
    args = ap.parse_args(argv)

    variants = [v for v in args.variants.split(",") if v]
    variants += [v for v, on in (("parapack", args.parapack), ("fixedlen", args.fixedlen)) if on]
    bad = set(variants) - set(BASE_VARIANTS) - set(LLM_VARIANTS)
    if bad:
        ap.error(f"unknown variants {bad}")
    try:
        check_frozen(current_config(), Path(args.frozen))
    except FrozenConfigMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    configure_llm_api(args.dataset)

    selection = json.loads(Path(args.selection).read_text())
    docs = load_docs(args.dataset, selection, args.domains.split(","))
    size = median_struct_tokens([d["markdown"] for d in load_docs(args.dataset, selection)]) \
        if "fixedlen" in variants else None
    if args.sample:
        docs = sorted(random.Random(2027).sample(docs, min(args.sample, len(docs))), key=lambda d: d["id"])
    if args.limit:
        docs = docs[: args.limit]
    todo = [d for d in docs if missing([d["id"]], variants, args.model_tag)]
    print(f"{args.dataset}/{args.model_tag}: {len(todo)}/{len(docs)} docs to build, variants={variants}, "
          f"num_ctx={NUM_CTX[args.dataset]}" + (f", fixedlen={size}" if size else ""), flush=True)
    b = Builder(args.dataset, args.model_tag, variants, fixedlen_size=size)
    res = b.run(todo, args.workers)
    print(f"SUMMARY done={res['done']} failed={len(res['failed'])} parse_failures={res['parse_failures']} "
          f"cost=${res['cost_usd']:.4f} budget_stop={res['budget_stop']}", flush=True)
    if res["budget_stop"]:
        return 3
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
