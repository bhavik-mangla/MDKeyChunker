"""Amendment 2 (H4): rolling keys and key-based merging. Retrieval exactly as evaluate_v2.

Systems: E, M (Stage 3 on E), RM (random adjacent merges, mean of 5 seeds per question),
SM (semantic adjacent merges), and, once built, E-noRK and M-noRK.

Family H4 (hybrid; plan v2 primary metric and margin per dataset; Holm over the 8 tests):
  H4a M - RM      two-sided sign-flip
  H4b M - SM      two-sided sign-flip
  H4c M - E       TOST (p_tost), two-sided also reported
  H4d E - E-noRK  TOST (p_tost), two-sided also reported
Exploratory (unadjusted): M-noRK - M, M - S+T, all H4 pairs under bm25/nomic/mxbai,
FreshStack per domain. Descriptive key statistics for E and E-noRK (Amendment 2 sec. C).

Writes results/v2/h4_{qasper,freshstack}_per_question.json, h4_stats.json, h4_summary.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_v2
import evaluate
import evaluate_v2 as ev

H4_LABELS = {
    "M": ("local", "merged", "meta"),
    "SM": ("local", "smerge", "meta"),
    "E-noRK": ("local", "enr_nork", "meta"),
    "M-noRK": ("local", "merged_nork", "meta"),
    "M-TP": ("local", "merged_tp", "meta"),
    **{f"RM{s}": ("local", f"rmerge_s{s}", "meta") for s in range(5)},
}
RM_SEEDS = [f"RM{s}" for s in range(5)]
H4 = [("M", "RM", None, "p_perm"), ("M", "SM", None, "p_perm"),
      ("M", "E", "margin", "p_tost"), ("E", "E-noRK", "margin", "p_tost")]
H7 = [("M", "M-TP")]  # Amendment 4: two-sided sign-flip, Holm over the 2 datasets
EXPLORATORY = [("M-TP", "E-noRK"), ("M-TP", "M-noRK"), ("M-noRK", "M"), ("M", "S+T"), ("M-noRK", "E-noRK")]
OUT = ev.OUT


def register() -> None:
    ev.SYSTEMS_V2.update(H4_LABELS)


def restrict(plan: dict, labels: list[str]) -> dict:
    return {**plan, "systems": [s for s in plan["systems"] if s in labels]}


def average_seeds(per: dict) -> None:
    """Replace RM0..RM4 by RM, the per-question mean over seeds."""
    for r, label, m in list(per):
        if label != "RM0":
            continue
        seeds = [per.get((r, s, m)) for s in RM_SEEDS]
        if any(s is None for s in seeds):
            continue
        per[(r, "RM", m)] = {k: [sum(x) / len(x) for x in zip(*(s[k] for s in seeds))]
                             for k in seeds[0] if all(k in s for s in seeds)}
    for k in [k for k in per if k[1] in RM_SEEDS]:
        del per[k]


def run_eval(retrievers: list[str], cache: Path) -> None:
    sel = json.loads((ev.V2 / "selection.json").read_text())
    wanted = ["E", "S+T", "M", "SM", "E-noRK", "M-noRK", "M-TP", *RM_SEEDS]
    papers = json.loads((ev.V2 / "qasper_v2.json").read_text())
    groups = {pid: [pid] for pid in sel["qasper"]["paper_ids"]}
    plan = restrict(h4_plan(groups, cache), wanted)
    res = ev.eval_qasper(papers, plan, cache, retrievers)
    average_seeds(res["per"])
    ev.dump_per_question(OUT / "h4_qasper_per_question.json", res, {"coverage": ev._coverage(plan)})
    print(f"qasper: systems={plan['systems']} dropped={len(plan['dropped'])}", flush=True)

    groups = {d: sel["freshstack"][d]["doc_ids"] for d in ev.FS_DOMAINS}
    plan = restrict(h4_plan(groups, cache), wanted)
    dqs = {}
    for d in ev.FS_DOMAINS:
        keep = set(sel["freshstack"][d]["qids"])
        dqs[d] = [q for q in json.loads((ev.V2 / f"freshstack_{d}.json").read_text())["questions"]
                  if q["qid"] in keep]
    res = ev.eval_freshstack(dqs, plan, cache, retrievers)
    average_seeds(res["per"])
    ev.dump_per_question(OUT / "h4_freshstack_per_question.json", res, {"coverage": ev._coverage(plan)})
    print(f"freshstack: systems={plan['systems']} dropped={len(plan['dropped'])}", flush=True)


def h4_plan(groups: dict, cache: Path) -> dict:
    """Paired completeness over the H4 arms only (a doc missing from E-noRK drops from every H4 arm)."""
    saved = dict(ev.SYSTEMS_V2)
    try:
        keep = {"E", "S+T", *H4_LABELS}
        for k in list(ev.SYSTEMS_V2):
            if k not in keep:
                del ev.SYSTEMS_V2[k]
        return ev.plan_systems(groups, cache)
    finally:
        ev.SYSTEMS_V2.clear()
        ev.SYSTEMS_V2.update(saved)


def specs() -> list[dict]:
    out = []
    for ds in ev.DATASETS:
        for a, b, margin, p_key in H4:
            out.append(ev._spec("h4", ds, ev.PRIMARY_RETRIEVER, ev.PRIMARY_METRIC[ds], a, b,
                                ev.MARGIN[ds] if margin else None, p_key))
    return out


def key_stats(variant: str) -> dict:
    """Amendment 2 sec. C, per dataset."""
    sel = json.loads((ev.V2 / "selection.json").read_text())
    ds_docs = {"qasper": sel["qasper"]["paper_ids"],
               "freshstack": [d for dom in ev.FS_DOMAINS for d in sel["freshstack"][dom]["doc_ids"]]}
    out = {}
    for ds, docs in ds_docs.items():
        n = reuse = sharing = distinct = merges = 0
        n_docs = 0
        for d in docs:
            p = build_v2.cache_path("local", variant, d)
            if not p.exists():
                continue
            n_docs += 1
            keys = [c.get("key", "") for c in json.loads(p.read_text())["chunks"]]
            seen: set = set()
            for k in keys:
                if k and k in seen:
                    reuse += 1
                seen.add(k)
            counts = {k: keys.count(k) for k in set(keys) if k}
            sharing += sum(v for v in counts.values() if v > 1)
            distinct += len(counts)
            n += len(keys)
            mp = build_v2.cache_path("local", {"enr": "merged", "enr_nork": "merged_nork", "enr_tp": "merged_tp"}[variant], d)
            if mp.exists():
                merges += len(keys) - len(json.loads(mp.read_text())["chunks"])
        out[ds] = {"docs": n_docs, "chunks": n, "key_reuse_rate": reuse / max(n, 1),
                   "chunks_sharing_a_key": sharing, "distinct_keys_per_doc": distinct / max(n_docs, 1),
                   "stage3_merges": merges}
    return out


def key_cost() -> dict:
    """Amendment 4 A: LLM calls/tokens for E (dictionary in-call) vs E-noRK + consolidation (TP)."""
    out = {}
    for ds, is_fs in (("qasper", False), ("freshstack", True)):
        row = {}
        for variant in ("enr", "enr_nork", "enr_tp"):
            tot = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "failed": 0}
            for p in (build_v2.CACHE_V2 / "local" / variant).glob("*.json"):
                if ("__" in p.stem) != is_fs:
                    continue
                st = json.loads(p.read_text())["stats"]
                for k in ("calls", "in_tokens", "out_tokens"):
                    tot[k] += st.get(k, 0) or 0
                tot["failed"] += int(bool(st.get("failed")))
            row[variant] = tot
        row["E_total"] = {k: row["enr"][k] for k in ("calls", "in_tokens", "out_tokens")}
        row["noRK_plus_TP_total"] = {k: row["enr_nork"][k] + row["enr_tp"][k] for k in ("calls", "in_tokens", "out_tokens")}
        out[ds] = row
    return out


def merged_length_stats() -> dict:
    """Share of M chunks over 512 mxbai-ish tokens is approximated with cl100k (descriptive only)."""
    enc = build_v2.ENC if hasattr(build_v2, "ENC") else None
    if enc is None:
        from build import ENC as enc  # noqa: N811
    out = {}
    for variant in ("enr", "merged"):
        lens = [len(enc.encode(c["text"])) for p in (build_v2.CACHE_V2 / "local" / variant).glob("*.json")
                for c in json.loads(p.read_text())["chunks"]]
        lens.sort()
        out[variant] = {"chunks": len(lens), "median_tokens": lens[len(lens) // 2],
                        "p95_tokens": lens[int(len(lens) * 0.95)],
                        "share_over_512": sum(x > 512 for x in lens) / len(lens)}
    return out


def analyze(n: int = 10_000) -> dict:
    data = {}
    for ds in ev.DATASETS:
        p = OUT / f"h4_{ds}_per_question.json"
        if p.exists():
            data[ds] = ev.load_per_question(p)
    rows = ev.run_family(specs(), data, n)
    h7 = ev.run_family([ev._spec("h7", ds, ev.PRIMARY_RETRIEVER, ev.PRIMARY_METRIC[ds], a, b, None, "p_perm")
                        for ds in ev.DATASETS for a, b in H7], data, n)
    confirmed = {(s["dataset"], s["retriever"], s["metric"], s["a"], s["b"]) for s in specs()}
    confirmed |= {(ds, ev.PRIMARY_RETRIEVER, ev.PRIMARY_METRIC[ds], a, b) for ds in ev.DATASETS for a, b in H7}
    expl = []
    pairs = [(a, b) for a, b, _, _ in H4] + H7 + EXPLORATORY
    for ds, res in data.items():
        per = res["per"]
        for r in ev.RETRIEVERS:
            for metric in sorted({k[2] for k in per}):
                for a, b in pairs:
                    if (r, a, metric) in per and (r, b, metric) in per and (ds, r, metric, a, b) not in confirmed:
                        margin = ev.MARGIN[ds] if metric == ev.PRIMARY_METRIC[ds] and (a, b) in (("M", "E"), ("E", "E-noRK")) else None
                        expl.append({"family": "exploratory", "dataset": ds, "retriever": r, "metric": metric,
                                     "a": a, "b": b, **ev.compare(per[(r, a, metric)], per[(r, b, metric)],
                                                                  margin=margin, strata=res.get("strata"), n=n)})
        if res.get("strata"):
            m = ev.PRIMARY_METRIC[ds]
            for dom in sorted(set(res["strata"].values())):
                for a, b in pairs:
                    ka, kb = ("hybrid", a, m), ("hybrid", b, m)
                    if ka in per and kb in per:
                        sa = {q: v for q, v in per[ka].items() if res["strata"].get(q) == dom}
                        sb = {q: v for q, v in per[kb].items() if res["strata"].get(q) == dom}
                        expl.append({"family": "exploratory", "dataset": f"{ds}/{dom}", "retriever": "hybrid",
                                     "metric": m, "a": a, "b": b, **ev.compare(sa, sb, n=n)})
    result = {"h4": rows, "h7": h7, "exploratory": expl, "systems": ev.system_means(data, n),
              "key_stats": {"E": key_stats("enr"), "E-noRK": key_stats("enr_nork"),
                            "E-TP": key_stats("enr_tp")},
              "lengths": merged_length_stats(), "key_cost": key_cost()}
    (OUT / "h4_stats.json").write_text(json.dumps(result, indent=1))
    (OUT / "h4_summary.txt").write_text(summary(result))
    return result


def _row(r: dict) -> str:
    head = f"  {r['dataset']:<20} {r['a']:>7} - {r['b']:<7} {r['retriever']:<7} {r['metric']:<11}"
    if r.get("status") != "ok":
        return head + " not available"
    s = (f"{r['diff']:+.4f} 90%[{r['ci90_bca'][0]:+.3f},{r['ci90_bca'][1]:+.3f}] "
         f"95%[{r['ci95_bca'][0]:+.3f},{r['ci95_bca'][1]:+.3f}] p_perm={r['p_perm']:.4f}")
    if r.get("p_tost") is not None:
        s += f" p_tost={r['p_tost']:.4f} +/-{r['margin']}"
    if "p_holm" in r:
        s += f" p_holm={r['p_holm']:.4f}{'*' if r['reject_holm'] else ''}"
    if "p_holm_perm" in r and r.get("p_key") != "p_perm":
        s += f" p_holm_perm={r['p_holm_perm']:.4f}"
    return f"{head} {s} -> {r.get('label', r.get('verdict'))}"


def summary(res: dict) -> str:
    L = ["Amendment 2 (H4) results. A - B, BCa CIs, cluster bootstrap.", "",
         "H4 family (hybrid; Holm over 8 tests; p_perm for H4a/b, p_tost for H4c/d)"]
    L += [_row(r) for r in res["h4"]]
    L += ["", "H7 family (Amendment 4; hybrid; Holm over 2 two-sided sign-flip tests)"]
    L += [_row(r) for r in res.get("h7", [])]
    L += ["", "System means (hybrid, primary metric)"]
    for r in res["systems"]:
        if r["retriever"] == "hybrid" and r["metric"] == ev.PRIMARY_METRIC[r["dataset"]]:
            L.append(f"  {r['dataset']:<10} {r['system']:<7} {r['mean']:.4f} "
                     f"[{r['ci95_pct'][0]:.3f},{r['ci95_pct'][1]:.3f}]")
    L += ["", "Key statistics (Amendment 2 sec. C)", json.dumps(res["key_stats"], indent=1),
          "", "Chunk lengths (cl100k tokens)", json.dumps(res["lengths"], indent=1),
          "", "Key cost (Amendment 4 A)", json.dumps(res.get("key_cost"), indent=1),
          "", "Exploratory (unadjusted), primary metric only"]
    L += [_row(r) for r in res["exploratory"]
          if r["metric"] == ev.PRIMARY_METRIC[r["dataset"].split("/")[0]]]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", default=",".join(ev.RETRIEVERS))
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    register()
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    if not args.stats_only:
        run_eval(args.retrievers.split(","), build_v2.CACHE_V2)
    analyze(args.boot)
    print((OUT / "h4_summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
