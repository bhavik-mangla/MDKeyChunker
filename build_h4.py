"""Amendment 2 (H4) merge arms, built from cached enrichment; no LLM calls.

  merged       MDKeyChunker Stage 3 (Restructurer, package defaults) on local/enr      (M)
  merged_nork  the same on local/enr_nork                                              (M-noRK)
  rmerge_s{k}  random adjacent merges, same count as M per doc, seeds 0-4               (RM)
  smerge       greedy adjacent merges by nomic cosine of chunk text, same count as M    (SM)

RM and SM use the same 3,000-char cap as Restructurer and the pilot's merge_group
(text joined in document order, title/summary from the first member, keyword and
question lists unioned). Output: data/v2/cache/local/<variant>/<doc>.json plus
data/v2/cache/local/h4_merge_counts.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "phase0"))
from a4_random_merge import CAP, merge_group, random_merge  # noqa: E402
from mdkeychunker.config import Config  # noqa: E402
from mdkeychunker.models import Chunk  # noqa: E402
from mdkeychunker.restructurer import Restructurer  # noqa: E402

import build_v2  # noqa: E402
import evaluate  # noqa: E402

LOCAL = build_v2.CACHE_V2 / "local"
SEEDS = range(5)


def to_chunks(dicts: list[dict]) -> list[Chunk]:
    return [Chunk(text=c["text"], section_title=c.get("section", ""), title=c.get("title", ""),
                  summary=c.get("summary", ""), keywords=list(c.get("keywords", [])),
                  questions=list(c.get("questions", [])), key=c.get("key", ""),
                  related_keys=list(c.get("related_keys", [])), entities=list(c.get("entities", [])),
                  start_line=i, end_line=i)
            for i, c in enumerate(dicts)]


def stage3(dicts: list[dict]) -> tuple[list[dict], int]:
    """Package Stage 3. Returns chunks and the number of keyless short chunks given orphan context."""
    chunks = to_chunks(dicts)
    orphans = sum(1 for c in chunks if not c.key and len(c.text) < Config().min_orphan_size)
    merged = Restructurer(Config(log_level="WARNING")).restructure(copy.deepcopy(chunks))
    return [build_v2.chunk_dict(c) for c in merged], orphans


def semantic_merge(chunks: list[dict], n_merges: int) -> tuple[list[dict], int]:
    """Greedy: merge the most similar adjacent pair (mean of member text embeddings) under the cap."""
    if n_merges <= 0 or len(chunks) < 2:
        return [merge_group(chunks, [i]) for i in range(len(chunks))], 0
    emb = evaluate.embed("nomic", [c["text"] for c in chunks], False)
    groups = [[i] for i in range(len(chunks))]
    size = lambda g: sum(len(chunks[i]["text"]) for i in g) + 2 * (len(g) - 1)  # noqa: E731

    def vec(g: list[int]) -> np.ndarray:
        v = emb[g].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    done = 0
    for _ in range(n_merges):
        best, best_sim = None, -2.0
        for a in range(len(groups) - 1):
            if size(groups[a]) + size(groups[a + 1]) + 2 > CAP:
                continue
            sim = float(vec(groups[a]) @ vec(groups[a + 1]))
            if sim > best_sim:
                best, best_sim = a, sim
        if best is None:
            break
        groups[best] = groups[best] + groups[best + 1]
        del groups[best + 1]
        done += 1
    return [merge_group(chunks, g) for g in groups], done


# Amendment 4 (H7): TopicGPT-style post-hoc consolidation of keys assigned without the dictionary.
def key_rules() -> str:
    from mdkeychunker.enricher import ENRICH_PROMPT
    line = next(ln for ln in ENRICH_PROMPT.splitlines() if ln.lstrip().startswith('"key":'))
    return line.strip().removeprefix('"key": "').rstrip('",')


TP_PROMPT = """You are a document analysis expert. The subtopic keys below were assigned independently to consecutive chunks of ONE Markdown document, listed in order of first use. Different chunks may have received different wordings for the same specific aspect.

Rules each key was meant to follow:
{rules}

**Keys:**
{keys}

Map every key to a canonical key. Two keys should map to the same canonical key ONLY if they name the EXACT same specific aspect and their chunks would make a coherent single piece when combined. Use one of the listed keys as the canonical key. Keep a key unchanged if no other key names the same aspect.

Return ONLY valid JSON: {{"<key>": "<canonical key>", ...}} with every listed key."""


def consolidate(chunks: list[dict], adapter) -> tuple[list[dict], dict]:
    keys = list(dict.fromkeys(c.get("key", "") for c in chunks if c.get("key")))
    mapping: dict = {}
    if len(keys) > 1:
        r = adapter.call_json(TP_PROMPT.format(rules=key_rules(), keys="\n".join(f"- {k}" for k in keys)))
        if adapter.fatal:
            raise adapter.fatal
        if isinstance(r, dict):
            mapping = {k: str(v).strip().lower() for k, v in r.items()
                       if k in keys and isinstance(v, str) and str(v).strip().lower() in keys}
    out = [{**c, "key": mapping.get(c.get("key", ""), c.get("key", ""))} for c in chunks]
    return out, {"keys_before": len(keys), "keys_after": len({c["key"] for c in out if c["key"]}),
                 "mapped": sum(1 for k, v in mapping.items() if k != v), "failed": len(keys) > 1 and not mapping
                 and bool(adapter.failed), **adapter.stats()}


def run_consolidation() -> None:
    import llm_api
    docs = sorted(p.stem for p in (LOCAL / "enr_nork").glob("*.json"))
    for ds in ("qasper", "freshstack"):
        build_v2.configure_llm_api(ds, ["enr"])
        mine = [d for d in docs if ("__" in d) == (ds == "freshstack")]
        for did in mine:
            if build_v2.cache_path("local", "merged_tp", did).exists():
                continue
            e = json.loads((LOCAL / "enr_nork" / f"{did}.json").read_text())["chunks"]
            ad = build_v2.LLMAdapter(build_v2.MODELS["local"], tag=f"tp-{ds}")
            remapped, st = consolidate(e, ad)
            m, _ = stage3(remapped)
            save("enr_tp", did, remapped, st)
            save("merged_tp", did, m, {**st, "n_merges": len(e) - len(m)})
        print(f"{ds}: consolidated {len(mine)} docs (num_ctx {llm_api.NUM_CTX})", flush=True)


def run_tp_controls() -> None:
    """Exploratory (post hoc, after the H7 result): are M-TP's merges better than random or
    most-similar adjacent merges of the same count on the same E-noRK base?"""
    short = 0
    for p in sorted((LOCAL / "merged_tp").glob("*.json")):
        did = p.stem
        e = json.loads((LOCAL / "enr_nork" / f"{did}.json").read_text())["chunks"]
        n = len(e) - len(json.loads(p.read_text())["chunks"])
        for s in SEEDS:
            rm, k = random_merge(e, n, s * 1000 + zlib.crc32(did.encode()) % 997, True)
            save(f"rmerge_tp_s{s}", did, rm, {"n_merges_target": n, "n_merges_done": k})
            short += k < n
        sm, k = semantic_merge(e, n)
        save("smerge_tp", did, sm, {"n_merges_target": n, "n_merges_done": k})
        short += k < n
    print(f"tp controls built; runs that made fewer merges than M-TP: {short}")


def save(variant: str, did: str, chunks: list[dict], stats: dict) -> None:
    p = build_v2.cache_path("local", variant, did)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"chunks": chunks, "stats": stats}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("enr", "enr_nork"), default="enr")
    ap.add_argument("--no-controls", action="store_true", help="only Stage 3 (used for enr_nork)")
    ap.add_argument("--consolidate", action="store_true", help="Amendment 4: M-TP from enr_nork")
    ap.add_argument("--tp-controls", action="store_true",
                    help="exploratory: RM/SM on enr_nork with merged_tp's merge counts")
    args = ap.parse_args(argv)
    if args.consolidate:
        run_consolidation()
        return 0
    if args.tp_controls:
        run_tp_controls()
        return 0
    out_variant = "merged" if args.source == "enr" else "merged_nork"
    counts_path = LOCAL / f"h4_merge_counts_{args.source}.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    docs = sorted(p.stem for p in (LOCAL / args.source).glob("*.json"))
    for i, did in enumerate(docs):
        e = json.loads((LOCAL / args.source / f"{did}.json").read_text())["chunks"]
        m, orphans = stage3(e)
        n = len(e) - len(m)
        save(out_variant, did, m, {"n_source": len(e), "n_merges": n, "orphans_with_context": orphans})
        row = {"n_source": len(e), "n_merged": len(m), "n_merges": n, "orphans_with_context": orphans}
        if not args.no_controls:
            row["rm_done"] = []
            for s in SEEDS:
                rm, k = random_merge(e, n, s * 1000 + zlib.crc32(did.encode()) % 997, True)
                save(f"rmerge_s{s}", did, rm, {"n_merges_target": n, "n_merges_done": k})
                row["rm_done"].append(k)
            sm, k = semantic_merge(e, n)
            save("smerge", did, sm, {"n_merges_target": n, "n_merges_done": k})
            row["sm_done"] = k
        counts[did] = row
        if i % 50 == 0:
            print(f"{i}/{len(docs)} {did} {row}", flush=True)
    counts_path.write_text(json.dumps(counts, indent=1))
    short = {k: sum(1 for r in counts.values() if r.get(k) is not None and
                    (min(r[k]) if isinstance(r[k], list) else r[k]) < r["n_merges"])
             for k in ("rm_done", "sm_done")}
    print(f"{len(docs)} docs; merges {sum(r['n_merges'] for r in counts.values())}; "
          f"orphans with context {sum(r['orphans_with_context'] for r in counts.values())}; "
          f"docs where a control made fewer merges than M: {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
