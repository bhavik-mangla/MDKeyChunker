"""Analysis 6: re-compute the v3 PILOT pre-registered comparisons in v2 style.

95% and 90% paired percentile CIs (10k cluster bootstrap), sign-flip permutation
p-values (10k), TOST at +-5 pts (Qasper rec@512t) / +-3 pts (FreshStack prec@1024t),
Holm within families. Reads results/*_per_question.json only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stats_pilot import analyse, holm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase0"
OUT.mkdir(parents=True, exist_ok=True)

COMPS = [("RQ1", "struct/text", "fixedtok/text"), ("RQ1", "struct/text", "fixed512/text"),
         ("RQ2", "enr_rk/meta", "struct/tc"), ("RQ3", "enr_rk/meta", "cr/cr"),
         ("RQ4b", "merged_rk/meta", "merged_nork/meta"), ("RQ4b", "merged_rk/meta", "enr_rk/meta")]
DATA = {"qasper": ("rec@512t", 0.05), "freshstack": ("prec@1024t", 0.03)}


def cell_rows(retrievers=("bm25", "hybrid", "nomic", "mxbai")):
    rows = []
    for ds, (metric, margin) in DATA.items():
        vals = json.loads((ROOT / "results" / f"{ds}_per_question.json").read_text())["values"]
        for rq, a, b in COMPS:
            for r in retrievers:
                ka, kb = f"{r}|{a}|{metric}", f"{r}|{b}|{metric}"
                if ka not in vals or kb not in vals:
                    continue
                res = analyse(vals[ka], vals[kb], margin)
                rows.append({"dataset": ds, "rq": rq, "a": a, "b": b, "retriever": r, "metric": metric, **res})
    return rows


def add_holm(rows, name, select):
    idx = [i for i, r in enumerate(rows) if select(r)]
    for key, out in (("p_perm", "holm_p_perm"), ("p_tost", "holm_p_tost")):
        adj = holm([rows[i][key] for i in idx])
        for i, a in zip(idx, adj):
            rows[i].setdefault(out, {})[name] = a
    return len(idx)


def main():
    rows = cell_rows()
    fams = {
        # every pre-registered cell on the v3 primary retrievers (BM25, hybrid), both datasets
        "v3_all_primary_cells": lambda r: r["retriever"] in ("bm25", "hybrid"),
        # v2 primary family: E-S+T and E-CR, hybrid, both datasets
        "v2_primary": lambda r: r["retriever"] == "hybrid" and r["rq"] in ("RQ2", "RQ3"),
        # v2 RQ1 family: S-F512c and S-F256t, hybrid, both datasets
        "v2_rq1": lambda r: r["retriever"] == "hybrid" and r["rq"] == "RQ1",
    }
    sizes = {n: add_holm(rows, n, f) for n, f in fams.items()}
    (OUT / "a6_stats.json").write_text(json.dumps({"families": sizes, "rows": rows}, indent=1))

    lines = ["# A6 v2-style statistics on v3 PILOT data", "",
             "diff in points (x100); CIs: percentile cluster bootstrap 10k; p: cluster sign-flip 10k; "
             "TOST margin +-5 (Qasper rec@512t) / +-3 (FreshStack prec@1024t).",
             f"Holm families: {sizes}", "",
             "| dataset | RQ | A - B | retr | diff | 95% CI | 90% CI | p_perm | Holm(v3 all) | Holm(v2 fam) | p_TOST | Holm TOST (v2 fam) | verdict (CI) | verdict (perm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h = r.get("holm_p_perm", {})
        ht = r.get("holm_p_tost", {})
        v2 = h.get("v2_primary", h.get("v2_rq1"))
        v2t = ht.get("v2_primary", ht.get("v2_rq1"))
        f = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(
            f"| {r['dataset']} | {r['rq']} | {r['a']} - {r['b']} | {r['retriever']} | {100*r['diff']:+.1f} | "
            f"[{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}] | [{100*r['ci90'][0]:+.1f}, {100*r['ci90'][1]:+.1f}] | "
            f"{r['p_perm']:.3f} | {f(h.get('v3_all_primary_cells'))} | {f(v2)} | {r['p_tost']:.3f} | {f(v2t)} | {r['verdict_ci']} | {r['verdict_perm']} |")
    (OUT / "a6_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
