"""Secondary retrieval arms of Amendment 1 section A, evaluated like evaluate_v2.

SECONDARY ANALYSIS: every row is labelled "secondary (Amendment 1 A)", with
unadjusted 95% CIs; nothing here enters a Holm family of plan v2.

Retrievers written to results/v2/extra/<dataset>_per_question.json
  hybrid          reference: BM25 + nomic, RRF k=60 (recomputed exactly as evaluate_v2)
  hybrid+rerank   hybrid top-30 reordered by Qwen3-Reranker-0.6B (tail unchanged)
  qwen3           qwen3-embedding:0.6b dense (Ollama /api/embed)
  hybrid-qwen3    BM25 + qwen3-embedding, RRF k=60
  jina            jina-embeddings-v3 dense on S, S+T, E, CR (plain), and LC-S
                  (late chunking: struct chunks pooled from whole-document token embeddings)

Metrics are evaluate.question_metrics (Qasper: rec@256/512/1024t, rec@k, hit@k;
cluster = paper) and evaluate_corpus.question_metrics (FreshStack: prec/rec@512/1024/2048t,
hit@k, mrr@10; cluster = question, strata = domain). Coverage (paired completeness)
is evaluate_v2.plan_systems. Statistics reuse evaluate_v2.compare / system_means.

Run with .venv-ml (torch, transformers, sentence-transformers):
  .venv-ml/bin/python evaluate_extra.py                      # all arms, both datasets
  .venv-ml/bin/python evaluate_extra.py --arms rerank --datasets qasper --limit 1 --out <tmp>
  .venv-ml/bin/python evaluate_extra.py --stats-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import build_v2
import evaluate
import evaluate_corpus
import evaluate_v2 as ev
import retrieval_extra as rx

OUT = ev.OUT / "extra"
ARMS = ("rerank", "qwen3", "lc")
JINA_SYSTEMS = ("S", "S+T", "E", "CR")
LC_LABEL = "LC-S"
LC_PAIRS = [(LC_LABEL, b) for b in JINA_SYSTEMS]
CROSS = [("hybrid+rerank", "hybrid"), ("hybrid-qwen3", "hybrid")]
FAMILY = "secondary (Amendment 1 A)"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"


class Arms:
    def __init__(self, arms: set[str], reranker: rx.Reranker | None = None, jina=None,
                 lc_overlap: int = 2048, log=print):
        self.arms = arms
        self.reranker = reranker
        self.jina = jina
        self.lc_overlap = lc_overlap
        self.lc_report = rx.LateChunkReport()
        self.log = log

    @property
    def needs_hybrid(self) -> bool:
        return bool(self.arms & {"rerank", "qwen3"})

    def jina_plain(self, texts: list[str], task: str) -> np.ndarray:
        return rx.cached_encode(texts, task, self.jina.encode)

    def orders(self, label: str, docs: list[str], queries: list[str]) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {}
        if self.needs_hybrid:
            base = ev.rank_all(docs, queries, ["bm25", "hybrid"])
            out["hybrid"] = base["hybrid"]
            if "rerank" in self.arms:
                out["hybrid+rerank"] = [self.reranker.rerank(o, q, docs) for o, q in zip(base["hybrid"], queries)]
            if "qwen3" in self.arms:
                q3 = rx.qwen3_orders(docs, queries)
                out["qwen3"] = q3
                out["hybrid-qwen3"] = [rx.rrf(a, b) for a, b in zip(base["bm25"], q3)]
        if "lc" in self.arms and label in JINA_SYSTEMS:
            out["jina"] = rx.dense_orders(self.jina_plain(docs, "retrieval.passage"),
                                          self.jina_plain(queries, "retrieval.query"))
        return out

    def lc_orders(self, doc_ids: list[str], md: dict[str, str], cache: Path,
                  queries: list[str]) -> tuple[list[dict], list[list[int]]]:
        chunks, embs = [], []
        for d in doc_ids:
            cs = ev.load_chunks("S", [d], cache)
            E, rep = rx.late_chunk_cached(self.jina, d, md[d], [c["text"] for c in cs], overlap=self.lc_overlap,
                                          plain_fn=lambda t: self.jina_plain(t, "retrieval.passage"))
            self.lc_report.add(rep)
            chunks += cs
            embs.append(E)
        Q = self.jina_plain(queries, "retrieval.query")
        return chunks, rx.dense_orders(np.vstack(embs), Q)


def _systems(plan: dict, only: list[str] | None) -> list[str]:
    return [s for s in plan["systems"] if not only or s in only]


def eval_qasper(papers: list[dict], plan: dict, cache: Path, arms: Arms, md: dict[str, str],
                systems: list[str] | None = None, limit: int | None = None) -> dict:
    per: dict = {}
    qids: dict = {}
    done = 0
    for p in papers:
        pid = p["id"]
        if pid not in plan["groups"]:
            continue
        qs = [q for q in p["questions"] if q["evidence"]]
        if not qs:
            continue
        if limit is not None and done >= limit:
            break
        done += 1
        qids[pid] = [q["qid"] for q in qs]
        queries = [q["question"] for q in qs]

        def record(r, label, chunks, orders):
            for q, order in zip(qs, orders):
                for m, v in evaluate.question_metrics(order, chunks, q["evidence"]).items():
                    per.setdefault((r, label, m), {}).setdefault(pid, []).append(v)

        for label in _systems(plan, systems):
            if label in ev.STRONG and pid not in plan["strong_groups"].get(label, []):
                continue
            chunks = ev.load_chunks(label, [pid], cache)
            docs = [evaluate.render(c, ev.SYSTEMS_V2[label][2]) for c in chunks]
            for r, orders in arms.orders(label, docs, queries).items():
                record(r, label, chunks, orders)
        if "lc" in arms.arms and "S" in plan["systems"]:
            chunks, orders = arms.lc_orders([pid], md, cache, queries)
            record("jina", LC_LABEL, chunks, orders)
        arms.log(f"qasper {done} {pid} q={len(qs)}"
                 + (f" rerank={arms.reranker.stats()}" if arms.reranker else ""))
    return {"per": per, "question_ids": qids}


def eval_freshstack(domain_qs: dict[str, list[dict]], plan: dict, cache: Path, arms: Arms, md: dict[str, str],
                    systems: list[str] | None = None) -> dict:
    per: dict = {}
    strata: dict = {}
    for dom, qs in domain_qs.items():
        if dom not in plan["groups"] or not qs:
            continue
        ids = plan["groups"][dom]
        golds = [set().union(*(evaluate.ngrams(evaluate.words(e["text"])) for e in q["evidence"])) for q in qs]
        queries = [q["question"] for q in qs]
        for q in qs:
            strata[q["qid"]] = dom

        def record(r, label, chunks, orders):
            for q, gold, order in zip(qs, golds, orders):
                for m, v in evaluate_corpus.question_metrics(order, chunks, gold).items():
                    per.setdefault((r, label, m), {})[q["qid"]] = [v]

        for label in _systems(plan, systems):
            if label in ev.STRONG and dom not in plan["strong_groups"].get(label, []):
                continue
            chunks = ev.load_chunks(label, ids, cache)
            docs = [evaluate.render(c, ev.SYSTEMS_V2[label][2]) for c in chunks]
            for r, orders in arms.orders(label, docs, queries).items():
                record(r, label, chunks, orders)
            arms.log(f"freshstack {dom} {label} chunks={len(chunks)}"
                     + (f" rerank={arms.reranker.stats()}" if arms.reranker else ""))
        if "lc" in arms.arms and "S" in plan["systems"]:
            chunks, orders = arms.lc_orders(ids, md, cache, queries)
            record("jina", LC_LABEL, chunks, orders)
    return {"per": per, "question_ids": {k: [k] for k in strata}, "strata": strata}


# --------------------------------------------------------------------------- statistics

def _pairs(labels: list[str]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(ev.H12 + ev.H12_STRONG + ev.H3 + ev.EXTRA_PAIRS + LC_PAIRS +
                              [(x, "S") for x in labels if x != "S"]))


def secondary_rows(data: dict, n: int) -> list[dict]:
    rows = []
    for ds, res in data.items():
        per, st = res["per"], res.get("strata")
        labels = sorted({k[1] for k in per})
        pairs = _pairs(labels)
        prim = ev.PRIMARY_METRIC[ds]
        for (r, a, metric) in sorted(per):
            # within retriever: planned comparisons and every system vs S
            for pa_, pb_ in pairs:
                if pa_ != a or (r, pb_, metric) not in per:
                    continue
                margin = ev.MARGIN[ds] if metric == prim and (a, pb_) in ev.H12 + ev.H12_STRONG else None
                rows.append({"family": FAMILY, "adjustment": "none", "dataset": ds, "retriever": r,
                             "metric": metric, "a": a, "b": pb_,
                             **ev.compare(per[(r, a, metric)], per[(r, pb_, metric)], margin=margin,
                                          strata=st, n=n)})
            # across retrievers, same system: what the reranker / qwen3 hybrid adds
            for r1, r0 in CROSS:
                if r == r1 and (r0, a, metric) in per:
                    rows.append({"family": FAMILY, "adjustment": "none", "dataset": ds,
                                 "retriever": f"{r1} vs {r0}", "metric": metric, "a": a, "b": a,
                                 **ev.compare(per[(r1, a, metric)], per[(r0, a, metric)], strata=st, n=n)})
        if st:  # FreshStack per domain, primary metric, planned pairs, every extra retriever
            for dom in sorted(set(st.values())):
                for r in sorted({k[0] for k in per}):
                    for a, b in ev.H12 + ev.H12_STRONG + ev.H3 + LC_PAIRS:
                        ka, kb = (r, a, prim), (r, b, prim)
                        if ka not in per or kb not in per:
                            continue
                        sa = {q: v for q, v in per[ka].items() if st.get(q) == dom}
                        sb = {q: v for q, v in per[kb].items() if st.get(q) == dom}
                        margin = ev.MARGIN[ds] if (a, b) in ev.H12 + ev.H12_STRONG else None
                        rows.append({"family": FAMILY, "adjustment": "none", "dataset": f"{ds}/{dom}",
                                     "retriever": r, "metric": prim, "a": a, "b": b,
                                     **ev.compare(sa, sb, margin=margin, n=n)})
    return rows


def analyze(out_dir: Path = OUT, n: int = 10_000, write: bool = True, meta: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    data = {ds: ev.load_per_question(out_dir / f"{ds}_per_question.json")
            for ds in ev.DATASETS if (out_dir / f"{ds}_per_question.json").exists()}
    meta_p = out_dir / "run_meta.json"
    result = {"label": "SECONDARY (Amendment 1 section A); unadjusted 95% CIs; not part of any Holm family",
              "rows": secondary_rows(data, n), "systems": ev.system_means(data, n),
              "settings": {"resamples": n, "margins_for_H1_H2_rows": ev.MARGIN, "rrf_k": rx.RRF_K,
                           "rerank_top": rx.RERANK_TOP, "reranker_instruction": rx.RERANK_INSTRUCTION,
                           "qwen3_query_prefix": rx.QWEN3_QUERY_PREFIX, "primary_metric": ev.PRIMARY_METRIC},
              "run_meta": meta or (json.loads(meta_p.read_text()) if meta_p.exists() else {})}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stats_extra.json").write_text(json.dumps(result, indent=1))
        (out_dir / "summary_extra.txt").write_text(summary_text(result))
    return result


def summary_text(result: dict) -> str:
    L = [result["label"], ""]
    for ds, prim in ev.PRIMARY_METRIC.items():
        sys_rows = [r for r in result["systems"] if r["dataset"] == ds and r["metric"] == prim]
        if not sys_rows:
            continue
        L.append(f"{ds}: system means, {prim} (95% percentile CI)")
        for r in sorted(sys_rows, key=lambda r: (r["retriever"], r["system"])):
            L.append(f"  {r['retriever']:14} {r['system']:10} {r['mean']:.4f} "
                     f"[{r['ci95_pct'][0]:.3f},{r['ci95_pct'][1]:.3f}] clusters={r['n_clusters']} n={r['n_obs']}")
        L.append("")
        L.append(f"{ds}: comparisons on {prim} (A - B, BCa; unadjusted)")
        for r in result["rows"]:
            if r["dataset"] == ds and r["metric"] == prim:
                L.append(f"  [{r['retriever']}]" + ev._fmt_row(r))
        L.append("")
    L.append("FreshStack per domain: see stats_extra.json rows with dataset 'freshstack/<domain>'.")
    rm = result.get("run_meta") or {}
    if rm:
        L.append("")
        L.append("Run meta: " + json.dumps({k: v for k, v in rm.items() if k != "coverage"}))
        for ds, c in (rm.get("coverage") or {}).items():
            L.append(f"  coverage {ds}: systems={c['systems']} dropped={len(c['dropped_docs'])}/{c['n_docs']}"
                     f" strong_groups={c['strong_groups']}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def merge_dump(path: Path, res: dict, extra: dict, merge: bool, log=print) -> None:
    """Write the per-question dump; with merge, keep (retriever, system, metric) series of an
    earlier run of other arms when it covered the same questions (new series win)."""
    if merge and path.exists():
        old = ev.load_per_question(path)
        if old["question_ids"] == res["question_ids"]:
            res = {**res, "per": {**old["per"], **res["per"]}}
        else:
            log(f"WARN {path.name}: question set differs from the earlier dump; not merging")
    ev.dump_per_question(path, res, extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="qasper,freshstack")
    ap.add_argument("--arms", default=",".join(ARMS), help="subset of rerank,qwen3,lc")
    ap.add_argument("--systems", default="", help="comma list of v2 labels (default: all available)")
    ap.add_argument("--fs-domains", default=",".join(ev.FS_DOMAINS))
    ap.add_argument("--limit", type=int, default=None, help="Qasper: first N papers (smoke tests)")
    ap.add_argument("--selection", default=str(ev.V2 / "selection.json"))
    ap.add_argument("--cache", default=str(build_v2.CACHE_V2))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA, help="for nomic (hybrid) and qwen3-embedding")
    ap.add_argument("--reranker", default=rx.QWEN3_RERANKER, help=f"fallback: {rx.BGE_RERANKER}")
    ap.add_argument("--rerank-batch", type=int, default=16)
    ap.add_argument("--rerank-max-length", type=int, default=2048)
    ap.add_argument("--device", default=None, help="mps/cpu (default: mps when available)")
    ap.add_argument("--lc-overlap", type=int, default=2048)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="merge into existing dumps (run arms separately, e.g. --arms qwen3 then --arms rerank)")
    ap.add_argument("--no-stats", action="store_true",
                    help="write per-question dumps and run_meta only; compute and print no outcome statistics "
                         "(smoke tests / blinded runs)")
    ap.add_argument("--no-frozen-check", action="store_true")
    args = ap.parse_args(argv)
    out, cache = Path(args.out), Path(args.cache)
    if not args.no_frozen_check:
        try:
            build_v2.check_frozen(build_v2.current_config(), build_v2.FROZEN, write=False, ignore_git=True)
        except build_v2.FrozenConfigMismatch as e:
            print(f"REFUSED: {e}", flush=True)
            return 2
    meta: dict = {}
    if not args.stats_only:
        arms_set = {a for a in args.arms.split(",") if a}
        assert arms_set <= set(ARMS), arms_set
        evaluate.OLLAMA = args.ollama_url
        rx.register_qwen3(args.ollama_url)
        reranker = (rx.Reranker(args.reranker, max_length=args.rerank_max_length, batch_size=args.rerank_batch,
                                device=args.device) if "rerank" in arms_set else None)
        jina = rx.JinaV3(device=args.device) if "lc" in arms_set else None
        arms = Arms(arms_set, reranker, jina, args.lc_overlap, log=lambda s: print(s, flush=True))
        systems = [s for s in args.systems.split(",") if s] or None
        sel = json.loads(Path(args.selection).read_text())
        t0 = time.time()
        coverage = {}
        if "qasper" in args.datasets:
            papers = json.loads((ev.V2 / "qasper_v2.json").read_text())
            md = {p["id"]: p["markdown"] for p in papers}
            plan = ev.plan_systems({pid: [pid] for pid in sel["qasper"]["paper_ids"]}, cache)
            res = eval_qasper(papers, plan, cache, arms, md, systems, args.limit)
            coverage["qasper"] = ev._coverage(plan)
            merge_dump(out / "qasper_per_question.json", res, {"coverage": coverage["qasper"]}, args.merge)
        if "freshstack" in args.datasets:
            doms = [d for d in args.fs_domains.split(",") if d]
            groups = {d: sel["freshstack"][d]["doc_ids"] for d in doms}
            plan = ev.plan_systems(groups, cache)
            md = {d["id"]: d["markdown"] for d in build_v2.load_docs("freshstack", sel, doms)}
            dqs = {}
            for d in doms:
                keep = set(sel["freshstack"][d]["qids"])
                dqs[d] = [q for q in json.loads((ev.V2 / f"freshstack_{d}.json").read_text())["questions"]
                          if q["qid"] in keep]
            res = eval_freshstack(dqs, plan, cache, arms, md, systems)
            coverage["freshstack"] = ev._coverage(plan)
            merge_dump(out / "freshstack_per_question.json", res, {"coverage": coverage["freshstack"]}, args.merge)
        meta = {"arms": sorted(arms_set), "seconds": round(time.time() - t0, 1), "ollama_url": args.ollama_url,
                "reranker": reranker.stats() if reranker else None,
                "late_chunking": arms.lc_report.as_dict() if jina else None,
                "jina_device": jina.device if jina else None, "lc_overlap": args.lc_overlap,
                "coverage": coverage}
        out.mkdir(parents=True, exist_ok=True)
        mp = out / "run_meta.json"
        runs = json.loads(mp.read_text()).get("runs", []) if args.merge and mp.exists() else []
        meta = {**meta, "runs": runs + [{k: v for k, v in meta.items() if k != "coverage"}]}
        mp.write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: v for k, v in meta.items() if k != "coverage"}), flush=True)
    if args.no_stats:
        return 0
    analyze(out, n=args.boot, meta=meta or None)
    print((out / "summary_extra.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
