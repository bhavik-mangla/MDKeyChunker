"""v2 retrieval evaluation and confirmatory statistics (ANALYSIS_PLAN_v2 sec. 3-6).

Retrieval
  Qasper: per paper (the paper's own chunks), cluster = paper.
  FreshStack: pooled per domain (all selected docs of the domain), cluster = question,
  bootstrap strata = domain.
  Retrievers: bm25, nomic, hybrid (RRF k=60 of bm25 + nomic; primary), mxbai (secondary).
  Metrics are the v3 ones (evaluate.question_metrics / evaluate_corpus.question_metrics,
  built on evaluate.retrieved_units / unit_grams).

Systems come from whatever caches exist under data/v2/cache/<model_tag>/<variant>.
Paired completeness (plan sec. 7): a document missing from any non-strong arm is
dropped from every arm and counted. Strong arms are evaluated on the papers (Qasper)
or whole domains (FreshStack) they cover, and compared on the common clusters only.

Statistics (stats.py), all paired, 10,000 resamples:
  primary   H1 E vs S+T, H2 E vs CR; local model, hybrid; each dataset. TOST with
            margins 0.05 (Qasper rec@512t) / 0.04 (FreshStack prec@1024t); BCa 90% and
            95% CIs, sign-flip p; Holm over the 4 p_tost.
  h3        S vs F512c, S vs F256t; hybrid; each dataset; Holm over the 4 sign-flip p.
  key_secondary  H1/H2 with the strong model where available, and Qasper answer F1
            (e2e_v2.py output) for H1/H2; Holm within the family on p_tost.
  exploratory    everything else; unadjusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

import build_v2
import evaluate
import evaluate_corpus
import stats

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
OUT = ROOT / "results" / "v2"

# label -> (model tag, variant, index rendering)
SYSTEMS_V2 = {
    "F512c": ("base", "fixed512", "text"),
    "F256t": ("base", "fixedtok", "text"),
    "S": ("base", "struct", "text"),
    "S+T": ("base", "struct", "tc"),
    "PP": ("base", "parapack", "text"),
    "FL": ("base", "fixedlen", "text"),
    "E": ("local", "enr", "meta"),
    "CR": ("local", "cr", "cr"),
    "E-strong": ("strong", "enr", "meta"),
    "CR-strong": ("strong", "cr", "cr"),
}
STRONG = ("E-strong", "CR-strong")
RETRIEVERS = ("bm25", "nomic", "hybrid", "mxbai")
PRIMARY_RETRIEVER = "hybrid"
PRIMARY_METRIC = {"qasper": "rec@512t", "freshstack": "prec@1024t"}
MARGIN = {"qasper": 0.05, "freshstack": 0.04}
F1_MARGIN = 0.05  # answer-F1 margin; not stated in the draft plan (see summary note)
DATASETS = ("qasper", "freshstack")
FS_DOMAINS = ("laravel", "angular", "yolo")

H12 = [("E", "S+T"), ("E", "CR")]
H12_STRONG = [("E-strong", "S+T"), ("E-strong", "CR-strong")]
H3 = [("S", "F512c"), ("S", "F256t")]
EXTRA_PAIRS = [("E-strong", "E"), ("CR-strong", "CR"), ("PP", "S"), ("FL", "F256t"), ("FL", "S"),
               ("S+T", "S"), ("CR", "S+T")]


# --------------------------------------------------------------------------- coverage

def has(label: str, doc_id: str, cache: Path) -> bool:
    tag, variant, _ = SYSTEMS_V2[label]
    return build_v2.cache_path(tag, variant, doc_id, cache).exists()


def plan_systems(groups: dict[str, list[str]], cache: Path) -> dict:
    """groups: retrieval unit -> doc ids (Qasper: paper -> [paper]; FreshStack: domain -> docs).

    Returns available systems, the doc set common to every non-strong arm, dropped
    docs, and for each strong arm the groups it fully covers (within the common set)."""
    all_docs = [d for ds in groups.values() for d in ds]
    avail = [s for s in SYSTEMS_V2 if any(has(s, d, cache) for d in all_docs)]
    core = [s for s in avail if s not in STRONG]
    common = {d for d in all_docs if all(has(s, d, cache) for s in core)}
    cgroups = {g: [d for d in ds if d in common] for g, ds in groups.items()}
    cgroups = {g: ds for g, ds in cgroups.items() if ds}
    strong_groups = {s: [g for g, ds in cgroups.items() if all(has(s, d, cache) for d in ds)]
                     for s in avail if s in STRONG}
    return {"systems": avail, "core": core, "groups": cgroups,
            "dropped": sorted(set(all_docs) - common), "n_docs": len(all_docs),
            "strong_groups": strong_groups}


def load_chunks(label: str, doc_ids: list[str], cache: Path) -> list[dict]:
    tag, variant, _ = SYSTEMS_V2[label]
    out = []
    for d in doc_ids:
        out += json.loads(build_v2.cache_path(tag, variant, d, cache).read_text())["chunks"]
    return out


# --------------------------------------------------------------------------- ranking

def rrf(a: list[int], b: list[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion, identical to evaluate.rank('hybrid')."""
    score: dict[int, float] = {}
    for r_ in (a, b):
        for pos, idx in enumerate(r_):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


def rank_all(docs: list[str], queries: list[str], retrievers: list[str],
             rank_fn: Callable = evaluate.rank) -> dict[str, list[list[int]]]:
    """Rankings per retriever; bm25/nomic are computed once and fused for hybrid."""
    need = set(retrievers) | ({"bm25", "nomic"} if "hybrid" in retrievers else set())
    base = {r: rank_fn(r, docs, queries) for r in ("bm25", "nomic", "mxbai") if r in need}
    if "hybrid" in need:
        base["hybrid"] = [rrf(a, b) for a, b in zip(base["bm25"], base["nomic"])]
    return {r: base[r] for r in retrievers}


# --------------------------------------------------------------------------- retrieval

def eval_qasper(papers: list[dict], plan: dict, cache: Path, retrievers: list[str],
                rank_fn: Callable = evaluate.rank) -> dict:
    per: dict = {}
    qids: dict = {}
    for p in papers:
        pid = p["id"]
        if pid not in plan["groups"]:
            continue
        qs = [q for q in p["questions"] if q["evidence"]]
        if not qs:
            continue
        qids[pid] = [q["qid"] for q in qs]
        for label in plan["systems"]:
            if label in STRONG and pid not in plan["strong_groups"].get(label, []):
                continue
            chunks = load_chunks(label, [pid], cache)
            docs = [evaluate.render(c, SYSTEMS_V2[label][2]) for c in chunks]
            orders = rank_all(docs, [q["question"] for q in qs], retrievers, rank_fn)
            for r in retrievers:
                for q, order in zip(qs, orders[r]):
                    for m, v in evaluate.question_metrics(order, chunks, q["evidence"]).items():
                        per.setdefault((r, label, m), {}).setdefault(pid, []).append(v)
    return {"per": per, "question_ids": qids}


def eval_freshstack(domain_qs: dict[str, list[dict]], plan: dict, cache: Path, retrievers: list[str],
                    rank_fn: Callable = evaluate.rank) -> dict:
    per: dict = {}
    strata: dict = {}
    for dom, qs in domain_qs.items():
        if dom not in plan["groups"]:
            continue
        ids = plan["groups"][dom]
        golds = [set().union(*(evaluate.ngrams(evaluate.words(e["text"])) for e in q["evidence"])) for q in qs]
        for q in qs:
            strata[q["qid"]] = dom
        for label in plan["systems"]:
            if label in STRONG and dom not in plan["strong_groups"].get(label, []):
                continue
            chunks = load_chunks(label, ids, cache)
            docs = [evaluate.render(c, SYSTEMS_V2[label][2]) for c in chunks]
            orders = rank_all(docs, [q["question"] for q in qs], retrievers, rank_fn)
            for r in retrievers:
                for q, gold, order in zip(qs, golds, orders[r]):
                    for m, v in evaluate_corpus.question_metrics(order, chunks, gold).items():
                        per.setdefault((r, label, m), {})[q["qid"]] = [v]
    return {"per": per, "question_ids": {k: [k] for k in strata}, "strata": strata}


def dump_per_question(path: Path, res: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"question_ids": res["question_ids"], "values": {"|".join(k): v for k, v in res["per"].items()}}
    if res.get("strata"):
        body["strata"] = res["strata"]
    path.write_text(json.dumps({**body, **(extra or {})}))


def load_per_question(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    per = {tuple(k.split("|")): v for k, v in raw["values"].items()}
    return {"per": per, "question_ids": raw.get("question_ids", {}), "strata": raw.get("strata")}


# --------------------------------------------------------------------------- statistics

def _ci(x) -> list[float]:
    return [float(x[0]), float(x[1])]


def compare(pa: dict, pb: dict, *, margin: float | None = None, strata: dict | None = None,
            n: int = 10_000, seed: int = 0) -> dict:
    """Paired A - B on the clusters both arms have (same per-cluster question counts)."""
    keys = [k for k in pa if k in pb and len(pa[k]) == len(pb[k])]
    if not keys:
        return {"status": "not_available"}
    d = {k: [x - y for x, y in zip(pa[k], pb[k])] for k in keys}
    st = {k: strata[k] for k in keys} if strata else None
    b = stats.cluster_bootstrap(d, n=n, seed=seed, strata=st)
    perm = stats.sign_flip_test(d, n=n, seed=seed)
    mean_a = float(np.mean([v for k in keys for v in pa[k]]))
    mean_b = float(np.mean([v for k in keys for v in pb[k]]))
    row = {"status": "ok", "mean_a": mean_a, "mean_b": mean_b, "diff": b["mean"],
           "n_clusters": b["n_clusters"], "n_obs": b["n_obs"],
           "ci90_bca": _ci(b["bca"][0.90]), "ci95_bca": _ci(b["bca"][0.95]),
           "ci90_pct": _ci(b["percentile"][0.90]), "ci95_pct": _ci(b["percentile"][0.95]),
           "p_perm": perm["p"], "perm_exact": perm["exact"],
           "excludes_zero_95": b["bca"][0.95][0] > 0 or b["bca"][0.95][1] < 0}
    if margin:
        t = stats.tost(d, margin=margin, boot=b)
        row.update(margin=margin, verdict=t["verdict"], label=t["label"], p_tost=t["p_tost"],
                   p_tost_t=t["p_tost_t"])
    else:
        row["verdict"] = "different" if row["excludes_zero_95"] else "no_difference_detected"
    return row


def _spec(family, ds, r, metric, a, b, margin, p_key):
    return {"family": family, "dataset": ds, "retriever": r, "metric": metric, "a": a, "b": b,
            "margin": margin, "p_key": p_key}


def family_specs() -> list[dict]:
    specs = []
    for ds in DATASETS:
        for a, b in H12:
            specs.append(_spec("primary", ds, PRIMARY_RETRIEVER, PRIMARY_METRIC[ds], a, b, MARGIN[ds], "p_tost"))
    for ds in DATASETS:
        for a, b in H3:
            specs.append(_spec("h3", ds, PRIMARY_RETRIEVER, PRIMARY_METRIC[ds], a, b, None, "p_perm"))
    for ds in DATASETS:
        for a, b in H12_STRONG:
            specs.append(_spec("key_secondary", ds, PRIMARY_RETRIEVER, PRIMARY_METRIC[ds], a, b, MARGIN[ds],
                               "p_tost"))
    for a, b in H12 + H12_STRONG:
        specs.append(_spec("key_secondary", "qasper", PRIMARY_RETRIEVER, "answer_f1", a, b, F1_MARGIN, "p_tost"))
    return specs


def run_family(specs: list[dict], data: dict, n: int) -> list[dict]:
    rows = []
    for s in specs:
        res = data.get(s["dataset"])
        pa = res["per"].get((s["retriever"], s["a"], s["metric"])) if res else None
        pb = res["per"].get((s["retriever"], s["b"], s["metric"])) if res else None
        if not pa or not pb:
            rows.append({**s, "status": "not_available"})
            continue
        rows.append({**s, **compare(pa, pb, margin=s["margin"], strata=res.get("strata"), n=n)})
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        h = stats.holm([r[r["p_key"]] for r in ok])
        for r, p, rej in zip(ok, h["p_adj"], h["reject"]):
            r.update(p_holm=p, reject_holm=rej, family_m=len(ok))
    return rows


def exploratory(data: dict, confirmed: set, n: int) -> list[dict]:
    rows = []
    for ds, res in data.items():
        per = res["per"]
        labels = sorted({k[1] for k in per})
        pairs = list(dict.fromkeys(H12 + H12_STRONG + H3 + EXTRA_PAIRS +
                                   [(x, "S") for x in labels if x != "S"]))
        for (r, a, metric) in sorted(per):
            for pa_, pb_ in pairs:
                if pa_ != a or (r, pb_, metric) not in per or (ds, r, metric, a, pb_) in confirmed:
                    continue
                margin = MARGIN[ds] if metric == PRIMARY_METRIC[ds] and (a, pb_) in H12 + H12_STRONG else None
                rows.append({"family": "exploratory", "dataset": ds, "retriever": r, "metric": metric,
                             "a": a, "b": pb_, **compare(per[(r, a, metric)], per[(r, pb_, metric)],
                                                         margin=margin, strata=res.get("strata"), n=n)})
        # FreshStack per domain, primary pairs (secondary per plan sec. 6)
        if res.get("strata"):
            for dom in sorted(set(res["strata"].values())):
                for a, b in H12 + H12_STRONG + H3:
                    k_a, k_b = (PRIMARY_RETRIEVER, a, PRIMARY_METRIC[ds]), (PRIMARY_RETRIEVER, b, PRIMARY_METRIC[ds])
                    if k_a not in per or k_b not in per:
                        continue
                    sub_a = {q: v for q, v in per[k_a].items() if res["strata"].get(q) == dom}
                    sub_b = {q: v for q, v in per[k_b].items() if res["strata"].get(q) == dom}
                    margin = MARGIN[ds] if (a, b) in H12 + H12_STRONG else None
                    rows.append({"family": "exploratory", "dataset": f"{ds}/{dom}", "retriever": PRIMARY_RETRIEVER,
                                 "metric": PRIMARY_METRIC[ds], "a": a, "b": b,
                                 **compare(sub_a, sub_b, margin=margin, n=n)})
    return rows


def system_means(data: dict, n: int) -> list[dict]:
    rows = []
    for ds, res in data.items():
        st = res.get("strata")
        for (r, label, metric), pp in sorted(res["per"].items()):
            b = stats.cluster_bootstrap(pp, n=n, seed=0, levels=(0.95,),
                                        strata={k: st[k] for k in pp} if st else None)
            rows.append({"dataset": ds, "retriever": r, "system": label, "metric": metric, "mean": b["mean"],
                         "ci95_pct": _ci(b["percentile"][0.95]), "n_clusters": b["n_clusters"], "n_obs": b["n_obs"]})
    return rows


def analyze(out_dir: Path = OUT, n: int = 10_000, write: bool = True) -> dict:
    """Stats over the per-question dumps (and the e2e answer-F1 dump when present)."""
    out_dir = Path(out_dir)
    data = {}
    for ds in DATASETS:
        p = out_dir / f"{ds}_per_question.json"
        if p.exists():
            data[ds] = load_per_question(p)
    e2e = out_dir / "qasper_e2e_per_question.json"
    if e2e.exists() and "qasper" in data:
        data["qasper"]["per"].update(load_per_question(e2e)["per"])
    specs = family_specs()
    fams = {f: run_family([s for s in specs if s["family"] == f], data, n)
            for f in ("primary", "h3", "key_secondary")}
    confirmed = {(s["dataset"], s["retriever"], s["metric"], s["a"], s["b"]) for s in specs}
    result = {**fams, "exploratory": exploratory(data, confirmed, n), "systems": system_means(data, n),
              "settings": {"resamples": n, "margins": MARGIN, "answer_f1_margin": F1_MARGIN,
                           "primary_metric": PRIMARY_METRIC, "primary_retriever": PRIMARY_RETRIEVER}}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stats.json").write_text(json.dumps(result, indent=1))
        (out_dir / "summary.txt").write_text(summary_text(result, data))
    return result


# --------------------------------------------------------------------------- summary

def _fmt_row(r: dict) -> str:
    if r.get("status") != "ok":
        return f"  {r['dataset']:18} {r['a']:>9} - {r['b']:9} {r['metric']:11} not available"
    ci90, ci95 = r["ci90_bca"], r["ci95_bca"]
    s = (f"  {r['dataset']:18} {r['a']:>9} - {r['b']:9} {r['metric']:11} {r['diff']:+.4f} "
         f"90%[{ci90[0]:+.3f},{ci90[1]:+.3f}] 95%[{ci95[0]:+.3f},{ci95[1]:+.3f}] p_perm={r['p_perm']:.4f}")
    if "p_tost" in r:
        s += f" p_tost={r['p_tost']:.4f} +/-{r['margin']}"
    if "p_holm" in r:
        s += f" p_holm={r['p_holm']:.4f}{'*' if r['reject_holm'] else ''}"
    return s + f" -> {r.get('label', r['verdict'])}"


def summary_text(result: dict, data: dict) -> str:
    L = ["v2 retrieval results (ANALYSIS_PLAN_v2). A - B, BCa CIs, cluster bootstrap "
         f"({result['settings']['resamples']} resamples).", ""]
    titles = {"primary": "PRIMARY family (H1, H2; local model; hybrid; TOST; Holm over p_tost)",
              "h3": "H3 family (structure vs fixed; hybrid; two-sided; Holm over sign-flip p)",
              "key_secondary": "KEY SECONDARY family (strong model H1/H2; Qasper answer F1; Holm over p_tost)"}
    for f, t in titles.items():
        L.append(t)
        L += [_fmt_row(r) for r in result[f]]
        L.append("")
    L.append(f"Note: the answer-F1 margin ({F1_MARGIN}) is not fixed in the draft plan; fix it before unblinding.")
    L.append("")
    L.append("System means (hybrid, primary metric; 95% percentile CI)")
    for r in result["systems"]:
        if r["retriever"] == PRIMARY_RETRIEVER and r["metric"] in (PRIMARY_METRIC.get(r["dataset"]), "answer_f1"):
            L.append(f"  {r['dataset']:10} {r['system']:10} {r['metric']:11} {r['mean']:.4f} "
                     f"[{r['ci95_pct'][0]:.3f},{r['ci95_pct'][1]:.3f}] clusters={r['n_clusters']} n={r['n_obs']}")
    L.append("")
    L.append("Exploratory (unadjusted): FreshStack per domain, hybrid, primary pairs")
    L += [_fmt_row(r) for r in result["exploratory"] if "/" in r["dataset"]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="qasper,freshstack")
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS))
    ap.add_argument("--selection", default=str(V2 / "selection.json"))
    ap.add_argument("--cache", default=str(build_v2.CACHE_V2))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--stats-only", action="store_true", help="recompute stats from existing dumps")
    ap.add_argument("--frozen", default=str(build_v2.FROZEN))
    args = ap.parse_args(argv)
    out, cache = Path(args.out), Path(args.cache)
    try:
        build_v2.check_frozen(build_v2.current_config(), Path(args.frozen), write=False, ignore_git=True)
    except build_v2.FrozenConfigMismatch as e:
        print(f"REFUSED: {e}", flush=True)
        return 2
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    retrievers = args.retrievers.split(",")
    if not args.stats_only:
        sel = json.loads(Path(args.selection).read_text())
        coverage = {}
        if "qasper" in args.datasets:
            papers = json.loads((V2 / "qasper_v2.json").read_text())
            plan = plan_systems({pid: [pid] for pid in sel["qasper"]["paper_ids"]}, cache)
            res = eval_qasper(papers, plan, cache, retrievers)
            coverage["qasper"] = _coverage(plan)
            dump_per_question(out / "qasper_per_question.json", res, {"coverage": coverage["qasper"]})
            print(f"qasper: {len(res['question_ids'])} papers, systems={plan['systems']}, "
                  f"dropped={len(plan['dropped'])}", flush=True)
        if "freshstack" in args.datasets:
            groups = {d: sel["freshstack"][d]["doc_ids"] for d in FS_DOMAINS}
            plan = plan_systems(groups, cache)
            dqs = {}
            for d in FS_DOMAINS:
                keep = set(sel["freshstack"][d]["qids"])
                dqs[d] = [q for q in json.loads((V2 / f"freshstack_{d}.json").read_text())["questions"]
                          if q["qid"] in keep]
            res = eval_freshstack(dqs, plan, cache, retrievers)
            coverage["freshstack"] = _coverage(plan)
            dump_per_question(out / "freshstack_per_question.json", res, {"coverage": coverage["freshstack"]})
            print(f"freshstack: {len(res['strata'])} questions, systems={plan['systems']}, "
                  f"dropped={len(plan['dropped'])}", flush=True)
        (out / "coverage.json").write_text(json.dumps(coverage, indent=1))
    result = analyze(out, n=args.boot)
    print((out / "summary.txt").read_text())
    return 0


def _coverage(plan: dict) -> dict:
    frac = len(plan["dropped"]) / max(plan["n_docs"], 1)
    return {"systems": plan["systems"], "dropped_docs": plan["dropped"], "n_docs": plan["n_docs"],
            "dropped_fraction": frac, "over_5pct": frac > 0.05,
            "strong_groups": {k: len(v) for k, v in plan["strong_groups"].items()}}


if __name__ == "__main__":
    raise SystemExit(main())
