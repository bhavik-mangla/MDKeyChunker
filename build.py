"""Build every chunk set for every paper, caching per (variant, paper) so runs resume.

Chunk sets:
  fixed512     512-character windows, no overlap (the original paper's Config A)
  fixedtok     256-token windows with 32-token overlap (common production default)
  struct       MDKeyChunker Stage 1 (structure-aware Markdown chunks)
  enr_rk       struct + single-call LLM enrichment WITH rolling keys (Stage 2)
  enr_nork     struct + enrichment with rolling keys hidden from the prompt (ablation A1)
  merged_rk    enr_rk  + key-based restructuring (Stage 3, the full pipeline)
  merged_nork  enr_nork + key-based restructuring
  cr           struct + Anthropic-style contextual-retrieval prefix written by the LLM
"""
import argparse
import os
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import tiktoken

from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config
from mdkeychunker.enricher import LLMEnricher
from mdkeychunker.restructurer import Restructurer

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache"
ENC = tiktoken.get_encoding("cl100k_base")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# One context size for every call: Ollama reloads the model whenever num_ctx changes.
# It must hold a whole document for the contextual-retrieval baseline.
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))

CR_PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class NoRollingKeysEnricher(LLMEnricher):
    """Ablation A1: the prompt never sees prior keys. Everything else is identical.

    Says "(not provided)" rather than the first-chunk placeholder, which would
    contradict the chunk position and previous summary in the same prompt.
    """

    def _format_rolling_keys(self) -> str:
        return "(not provided)"


class OllamaJSON:
    """Drop-in for LLMClient.call_json via Ollama's native API.

    The native API lets us turn off qwen3's thinking mode and read exact token
    counts. Records calls, wall time, and tokens for the cost table.
    """

    def __init__(self, model: str):
        self.model = model
        self.calls = 0
        self.seconds = 0.0
        self.in_tok = 0
        self.out_tok = 0
        self.failures = 0

    def call_json(self, prompt: str, max_tokens: int = 1000):
        t = time.time()
        try:
            r = post_with_retry(f"{OLLAMA}/api/chat", {
                "model": self.model, "stream": False, "format": "json", "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens, "num_ctx": NUM_CTX},
            })
        except Exception:
            self.failures += 1  # the enricher swallows this; build_paper refuses to cache
            raise
        self.seconds += time.time() - t
        self.calls += 1
        self.in_tok += r.get("prompt_eval_count", 0)
        self.out_tok += r.get("eval_count", 0)
        try:
            return json.loads(r["message"]["content"])
        except (KeyError, json.JSONDecodeError):
            self.failures += 1
            return None

    def stats(self) -> dict:
        return {"model": self.model, "calls": self.calls, "seconds": round(self.seconds, 1),
                "in_tokens": self.in_tok, "out_tokens": self.out_tok}


def post_with_retry(url: str, payload: dict, tries: int = 6) -> dict:
    """Ride out a restarting Ollama server instead of silently losing the call."""
    for attempt in range(tries):
        try:
            return requests.post(url, json=payload, timeout=900).json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def chunk_dict(c) -> dict:
    return {"text": c.text, "section": c.section_title, "title": c.title, "summary": c.summary,
            "keywords": c.keywords, "questions": c.questions, "key": c.key,
            "related_keys": c.related_keys, "entities": c.entities}


def fixed_chars(md: str, size: int = 512) -> list[dict]:
    return [{"text": md[i:i + size]} for i in range(0, len(md), size)]


def fixed_tokens(md: str, size: int = 256, overlap: int = 32) -> list[dict]:
    toks = ENC.encode(md)
    step = size - overlap
    return [{"text": ENC.decode(toks[i:i + size])} for i in range(0, max(len(toks) - overlap, 1), step)]


def contextual_prefixes(md: str, chunks: list[dict], model: str) -> tuple[list[dict], dict]:
    # Document first so Ollama reuses the KV cache for the shared prefix across chunks.
    out, t0, in_tok, out_tok = [], time.time(), 0, 0
    for c in chunks:
        prompt = CR_PROMPT.format(doc=md, chunk=c["text"])
        r = post_with_retry(f"{OLLAMA}/api/chat", {
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "options": {"num_ctx": NUM_CTX, "temperature": 0, "seed": 0, "num_predict": 120},
        })
        ctx = r["message"]["content"].strip()
        in_tok += r.get("prompt_eval_count", 0)
        out_tok += r.get("eval_count", 0)
        out.append({**c, "context": ctx})
    return out, {"model": model, "calls": len(chunks), "seconds": round(time.time() - t0, 1),
                 "in_tokens_evaluated": in_tok, "out_tokens": out_tok}


def build_paper(paper: dict, cfg: Config, variants: set[str]) -> None:
    pid, md = paper["id"], paper["markdown"]

    def cached(name: str):
        p = CACHE / name / f"{pid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save(name: str, chunks: list[dict], stats: dict | None = None) -> None:
        p = CACHE / name / f"{pid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"chunks": chunks, "stats": stats or {}}))

    if "fixed512" in variants and not cached("fixed512"):
        save("fixed512", fixed_chars(md))
    if "fixedtok" in variants and not cached("fixedtok"):
        save("fixedtok", fixed_tokens(md))

    struct = MarkdownChunker(cfg).chunk(md)
    if not cached("struct"):
        save("struct", [chunk_dict(c) for c in struct])

    for rk, cls in (("rk", LLMEnricher), ("nork", NoRollingKeysEnricher)):
        name = f"enr_{rk}"
        if name not in variants:
            continue
        hit = cached(name)
        if hit is None:
            llm = OllamaJSON(cfg.llm_model)
            enricher = cls(llm)  # type: ignore[arg-type]
            enriched = enricher.enrich_chunks(copy.deepcopy(struct))
            if llm.failures:
                raise RuntimeError(f"{pid} {name}: {llm.failures} failed LLM calls; not caching")
            save(name, [chunk_dict(c) for c in enriched], llm.stats())
        else:
            enriched = None
        if f"merged_{rk}" in variants and not cached(f"merged_{rk}"):
            if enriched is None:
                from mdkeychunker.models import Chunk
                enriched = [Chunk(text=c["text"], section_title=c["section"], title=c["title"],
                                  summary=c["summary"], keywords=c["keywords"],
                                  questions=c["questions"], key=c["key"],
                                  related_keys=c["related_keys"], entities=c["entities"])
                            for c in cached(name)["chunks"]]
            merged = Restructurer(cfg).restructure(copy.deepcopy(enriched))
            save(f"merged_{rk}", [chunk_dict(c) for c in merged])

    if "cr" in variants and not cached("cr"):
        chunks, stats = contextual_prefixes(md, cached("struct")["chunks"], cfg.llm_model)
        save("cr", chunks, stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(ROOT / "data" / "subset.json"))
    ap.add_argument("--papers", type=int, default=0, help="limit (0 = all)")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="fixed512,fixedtok,struct,enr_rk,enr_nork,merged_rk,merged_nork,cr")
    args = ap.parse_args()

    papers = json.loads(Path(args.subset).read_text())
    if args.papers:
        papers = papers[: args.papers]
    cfg = Config(llm_provider="openai_compatible", llm_base_url=f"{OLLAMA}/v1",
                 llm_model=args.model, log_level="WARNING")
    variants = set(args.variants.split(","))

    def run(p: dict) -> None:
        t = time.time()
        try:
            build_paper(p, cfg, variants)
            print(f"{p['id']} done in {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # keep other papers going; a rerun resumes this one
            print(f"{p['id']} FAILED: {e!r}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, papers))


if __name__ == "__main__":
    main()
