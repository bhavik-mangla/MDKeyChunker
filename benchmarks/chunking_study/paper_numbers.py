"""Emit every number the paper cites as LaTeX macros, straight from the result files.

Usage:  .venv/bin/python paper_numbers.py [--out ../MDKeyChunker-paper/numbers.tex]
In the paper: \\input{numbers} then \\R{key}, e.g. \\R{H1-qasper-hyb}. Differences are in
points (x100), one decimal; CIs in brackets. A missing source prints "??" so it shows in the PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
V2 = ROOT / "results" / "v2"
M: dict[str, str] = {}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pt(x: float, sign: bool = True) -> str:
    v = 100 * x
    s = f"{v:+.1f}" if sign else f"{v:.1f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(lo_hi) -> str:
    return f"[{pt(lo_hi[0])}, {pt(lo_hi[1])}]"


VERDICT = {"equivalent": "equiv.", "different": "diff.", "inconclusive": "incon.",
           "no_difference_detected": "n.d.", "different_but_equivalent": "diff., equiv."}


def put(key: str, value: str) -> None:
    M[key] = value


def row(prefix: str, r: dict | None, level: str = "90") -> None:
    if not r or r.get("status") != "ok":
        for suf in ("", "-d", "-ci", "-v", "-p"):
            put(prefix + suf, "--")
        return
    c = r[f"ci{level}_bca"]
    put(prefix + "-d", pt(r["diff"]))
    put(prefix + "-ci", ci(c))
    if "ci95_bca" in r:
        put(prefix + "-ci95", ci(r["ci95_bca"]))
    put(prefix, f"{pt(r['diff'])} {ci(c)}")
    put(prefix + "-v", VERDICT.get(r.get("label", r["verdict"]), r.get("label", r["verdict"])))
    p = r.get("p_holm")
    put(prefix + "-p", f"{p:.3f}" if p is not None else "--")
    if "p_holm_perm" in r:
        put(prefix + "-pperm", f"{r['p_holm_perm']:.3f}")


def find(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def neg(r: dict | None) -> dict | None:
    """B - A from an A - B row (bootstrap CIs flip)."""
    if not r or r.get("status") != "ok":
        return r
    out = dict(r)
    out["diff"] = -r["diff"]
    for k in ("ci90_bca", "ci95_bca"):
        out[k] = [-r[k][1], -r[k][0]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "MDKeyChunker-paper" / "numbers.tex"))
    args = ap.parse_args()
    ds_short = {"qasper": "qasper", "freshstack": "fs"}
    pm = {"qasper": "rec@512t", "freshstack": "prec@1024t"}

    # ---------------------------------------------------------------- confirmatory (plan v2)
    s = load(V2 / "stats.json")
    names = {("E", "S+T"): "H1", ("E", "CR"): "H2", ("E-strong", "S+T"): "H1s", ("E-strong", "CR-strong"): "H2s",
             ("S", "F512c"): "H3a", ("S", "F256t"): "H3b"}
    for fam in ("primary", "key_secondary", "h3"):
        for r in s[fam]:
            if r["metric"] not in pm.values():
                continue
            key = f"{names[(r['a'], r['b'])]}-{ds_short[r['dataset']]}"
            row(f"{key}-hyb", r, "95" if fam == "h3" else "90")
            bm = find(s["exploratory"], dataset=r["dataset"], retriever="bm25", metric=r["metric"], a=r["a"], b=r["b"])
            put(f"{key}-bm25", pt(bm["diff"]) if bm and bm.get("status") == "ok" else "--")
            if bm and bm.get("status") == "ok":
                put(f"{key}-bm25ci", ci(bm["ci95_bca"]))
            for ret in ("nomic", "mxbai"):
                dr = find(s["exploratory"], dataset=r["dataset"], retriever=ret, metric=r["metric"], a=r["a"], b=r["b"])
                row(f"{key}-{ret}", dr, "95")
    for r in s["key_secondary"]:
        if r["metric"] == "answer_f1":
            row(f"F1-{'H1' if r['b'] == 'S+T' else 'H2'}", r)
    for r in s["exploratory"]:
        if r["metric"] == "answer_f1" and r["retriever"] == "hybrid" and r["a"] == "E-strong" and r["b"] in ("S+T", "CR-strong"):
            row(f"F1-{'H1s' if r['b'] == 'S+T' else 'H2s'}", r, "95")
    for ds in ("qasper", "freshstack"):
        d = ds_short[ds]
        for a, b, k in (("PP", "S", "SP"), ("FL", "S", "SFL"), ("S+T", "S", "STS"), ("CR", "S+T", "CRST"),
                        ("E-strong", "E", "EsE")):
            r = find(s["exploratory"], dataset=ds, retriever="hybrid", metric=pm[ds], a=a, b=b)
            row(f"{k}-{d}", neg(r) if k in ("SP", "SFL") else r, "95")
        for r in s["systems"]:
            if r["dataset"] == ds and r["retriever"] == "hybrid" and r["metric"] in (pm[ds], "answer_f1"):
                tag = "f1" if r["metric"] == "answer_f1" else "m"
                put(f"{tag}-{d}-{r['system']}", pt(r["mean"], sign=False))
            elif r["dataset"] == ds and r["metric"] == pm[ds]:
                put(f"m-{d}-{r['retriever']}-{r['system']}", pt(r["mean"], sign=False))
        for dom in ("laravel", "angular", "yolo"):
            for a, b, k in (("E", "S+T", "H1"), ("E", "CR", "H2"), ("S", "F256t", "H3b")):
                r = find(s["exploratory"], dataset=f"freshstack/{dom}", retriever="hybrid", a=a, b=b,
                         metric=pm["freshstack"])
                row(f"{k}-{dom}", r, "95")

    # ---------------------------------------------------------------- H4, H7 (Amendments 2, 4)
    h4 = load(V2 / "h4_stats.json")
    if h4:
        k4 = {("M", "RM"): "H4a", ("M", "SM"): "H4b", ("M", "E"): "H4c", ("E", "E-noRK"): "H4d"}
        for r in h4["h4"]:
            row(f"{k4[(r['a'], r['b'])]}-{ds_short[r['dataset']]}", r, "95" if r["p_key"] == "p_perm" else "90")
        for r in h4["h7"]:
            row(f"H7-{ds_short[r['dataset']]}", r, "95")
        for r in h4["systems"]:
            if r["retriever"] == "hybrid" and r["metric"] == pm[r["dataset"]]:
                put(f"m-{ds_short[r['dataset']]}-{r['system']}", pt(r["mean"], sign=False))
        for r in h4["exploratory"]:  # merged set minus its own unmerged base (Table "dial")
            if r["retriever"] == "hybrid" and r["dataset"] in ds_short and r["metric"] == pm[r["dataset"]] \
                    and (r["a"], r["b"]) in (("M-noRK", "E-noRK"), ("M-TP", "E-noRK")):
                row(f"base-{r['a']}-{ds_short[r['dataset']]}", r, "95")
        for r in h4["h4"]:
            if (r["a"], r["b"]) == ("M", "E"):
                row(f"base-M-{ds_short[r['dataset']]}", r, "95")
        for arm, st in h4["key_stats"].items():
            for ds, v in st.items():
                d = ds_short[ds]
                put(f"reuse-{arm}-{d}", f"{100 * v['key_reuse_rate']:.1f}")
                put(f"keys-{arm}-{d}", f"{v['distinct_keys_per_doc']:.1f}")
                put(f"merges-{arm}-{d}", f"{v['stage3_merges']:,}")
        for v, st in h4["lengths"].items():
            put(f"len-{v}-median", str(st["median_tokens"]))
            put(f"len-{v}-over512", f"{100 * st['share_over_512']:.1f}")
        for ds, row_ in (h4.get("key_cost") or {}).items():
            d = ds_short[ds]
            e, n = row_["E_total"], row_["noRK_plus_TP_total"]
            put(f"cost-dict-in-{d}", f"{100 * (row_['enr']['in_tokens'] / row_['enr_nork']['in_tokens'] - 1):.0f}")
            put(f"cost-tp-in-{d}", f"{100 * (row_['enr_tp']['in_tokens'] / row_['enr_nork']['in_tokens']):.1f}")
            put(f"cost-E-vs-TP-in-{d}", f"{100 * abs(n['in_tokens'] / e['in_tokens'] - 1):.0f}")  # text says "fewer"
            assert n["in_tokens"] < e["in_tokens"], "text assumes TP uses fewer input tokens"
            put(f"tp-failed-{d}", str(row_["enr_tp"]["failed"]))

    # ---------------------------------------------------------------- dial controls (exploratory, post hoc)
    dl = load(V2 / "dial_stats.json")
    if dl:
        for r in dl["rows"]:
            if r["retriever"] == "hybrid" and r.get("status") == "ok":
                suf = "-rec" if r["metric"] == "rec@1024t" else ""
                row(f"dial-{r['a']}-{r['b']}-{ds_short[r['dataset']]}{suf}", r, "95")
        c = dl["merge_counts"]
        put("dial-short", str(sum(v["target"] - v["done"] for v in c.values())))

    # ---------------------------------------------------------------- H5, H6 (Amendment 3)
    h5 = load(V2 / "h5_stats.json")
    if h5:
        for r in h5["h5"]:
            row(f"{'H5a' if r['b'] == 'E' else 'H5b'}-{ds_short[r['dataset']]}", r, "95")
        for r in h5["h6"]:
            row("H6", r)
        for r in h5["systems"]:
            if r["retriever"] == "hybrid" and r["metric"] == pm[r["dataset"]]:
                put(f"m-{ds_short[r['dataset']]}-{r['system']}", pt(r["mean"], sign=False))
        for r in h5["exploratory"]:
            if r["retriever"] == "hybrid" and r["dataset"] in ds_short and r["metric"] == pm[r["dataset"]] \
                    and (r["a"], r["b"]) in (("E+SX", "E"), ("E(meta+key)", "E"), ("E+KX", "E+SX")):
                row(f"{r['a']}-{r['b']}-{ds_short[r['dataset']]}".replace("+", "").replace("(", "").replace(")", ""), r, "95")
            if r["retriever"] == "hybrid" and r["dataset"] in ds_short and r["metric"] == pm[r["dataset"]] \
                    and (r["a"], r["b"]) == ("E+AX", "E"):
                row(f"EAX-E-{ds_short[r['dataset']]}", r, "95")
        for ds, v in (h5.get("key_links") or {}).items():
            d = ds_short[ds]
            for k, tag in (("chunks_with_links", "all"), ("retrieved_with_links", "ret")):
                if v.get(k):
                    put(f"links-{tag}-{d}", f"{100 * v[k][0] / max(v[k][1], 1):.0f}")
        pc = h5["pf_cost"]
        e, p = pc["enr"], pc["perfield"]
        put("pf-calls-ratio", f"{p['calls'] / e['calls']:.1f}")
        put("pf-in-ratio", f"{p['in_tokens'] / e['in_tokens']:.2f}")
        put("pf-out-ratio", f"{p['out_tokens'] / e['out_tokens']:.2f}")

    # ---------------------------------------------------------------- FreshStack answers (Amendment 1 B)
    fe = load(V2 / "extra" / "freshstack_e2e.json")
    if fe:
        for sname, v in fe["means"].items():
            put(f"nug-{sname}", pt(v["strict"]["mean"], sign=False))
        for c in fe["comparisons"]:
            if c["metric"] == "nugget_strict" and c.get("status") == "ok":
                row(f"nug-{c['a']}-{c['b']}".replace("+", "p"), c, "95")
        ctl = fe["controls"]
        put("ctl-pos", f"{100 * ctl['positive_strict_recall']:.1f}")
        put("ctl-neg", f"{100 * ctl['negative_strict_recall']:.1f}")
        put("judge-failed", str(len(fe["meta"].get("judge_failed_rows", []))))

    # ---------------------------------------------------------------- scaling (Amendment 1 C)
    sc = load(V2 / "extra" / "scaling_qasper.json")
    if sc:
        put("scale-n", str(len(sc["papers_used"])))
        for k in ("E", "E-strong", "E-qwen235b", "S+T"):
            put(f"scale-{k}", pt(sc["means"][f"hybrid|{k}|rec@512t"], sign=False))
        for c in sc["comparisons"]:
            if c["retriever"] == "hybrid" and c.get("metric", "rec@512t") == "rec@512t":
                row(f"scale-{c['a']}-{c['b']}", c, "95")

    # ---------------------------------------------------------------- cost (server token counters)
    cache = ROOT / "data" / "v2" / "cache"
    for tag in ("local", "strong"):
        for ds, is_fs in (("qasper", False), ("fs", True)):
            tot = {}
            for v in ("enr", "cr"):
                t = {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
                for p in (cache / tag / v).glob("*.json"):
                    if ("__" in p.stem) != is_fs:
                        continue
                    st = json.loads(p.read_text())["stats"]
                    t["calls"] += st.get("calls", 0) or 0
                    t["in"] += st.get("in_tokens", 0) or 0
                    t["out"] += st.get("out_tokens", 0) or 0
                    t["usd"] += st.get("cost_usd", 0) or 0
                tot[v] = t
                if t["calls"]:
                    put(f"tok-{tag}-{v}-{ds}-in", f"{t['in'] / 1e6:.1f}M")
                    put(f"tok-{tag}-{v}-{ds}-percall", f"{t['in'] / t['calls']:,.0f}")
                    put(f"tok-{tag}-{v}-{ds}-outpercall", f"{t['out'] / t['calls']:,.0f}")
                    if tag == "strong":
                        put(f"usd-{v}-{ds}", f"{t['usd']:.2f}")
                        put(f"usdpercall-{v}-{ds}", f"{1e6 * t['usd'] / t['calls']:.0f}")  # micro-dollars
            if tot["enr"]["calls"] and tot["cr"]["calls"]:
                # per call, since CR-strong covers a 150-paper sample
                r = (tot["cr"]["in"] / tot["cr"]["calls"]) / (tot["enr"]["in"] / tot["enr"]["calls"])
                put(f"crratio-{tag}-{ds}", f"{r:.1f}")

    # ---------------------------------------------------------------- secondary retrievers (Amendment 1 A)
    sx = load(V2 / "extra" / "stats_extra.json") or {"rows": [], "systems": []}
    for ds in ("qasper", "freshstack"):
        d = ds_short[ds]
        for r_, a, b, k in (("hybrid+rerank", "E", "S+T", "rrH1"), ("hybrid+rerank", "E", "CR", "rrH2"),
                            ("qwen3", "E", "S+T", "q3H1"), ("hybrid-qwen3", "E", "S+T", "hq3H1"),
                            ("jina", "LC-S", "S", "lcS"), ("jina", "LC-S", "E", "lcE")):
            r = find(sx["rows"], dataset=ds, retriever=r_, metric=pm[ds], a=a, b=b)
            row(f"{k}-{d}", neg(r) if k == "lcE" else r, "95")
        for r in sx["systems"]:
            if r["dataset"] == ds and r["metric"] == pm[ds]:
                put(f"m-{d}-{r['retriever']}-{r['system']}", pt(r["mean"], sign=False))

    lcs = load(V2 / "extra" / "lc_similarity.json") or {}
    for ds, r in lcs.items():
        for k in ("plain", "late"):
            put(f"lcsim-{ds_short[ds]}-{k}", f"{r[k]:.2f}")
        put(f"lcsim-{ds_short[ds]}-n", str(r["n_docs"]))

    # ---------------------------------------------------------------- spend
    total = 0.0
    led = ROOT / "results" / "spend_ledger.jsonl"
    if led.exists():
        for line in led.read_text().splitlines():
            total += json.loads(line).get("cost_usd") or 0
    put("spend", f"{total:.2f}")

    write_grids(Path(args.out).with_name("grids.tex"), pm)

    lines = ["% Generated by MDKeyChunker-eval/paper_numbers.py; do not edit by hand.",
             "\\makeatletter", "\\providecommand{\\R}[1]{\\@ifundefined{r@#1}{\\textbf{??#1}}{\\csname r@#1\\endcsname}}"]
    for k in sorted(M):
        lines.append(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{M[k]}}}")
    lines.append("\\makeatother")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(M)} macros -> {args.out}")


# ---------------------------------------------------------------------------- full result grids

GRID_RET = [("bm25", "BM25"), ("nomic", "nomic"), ("mxbai", "mxbai"), ("hybrid", "Hybrid"), ("qwen3", "qwen3"),
            ("hybrid-qwen3", "BM25+qwen3"), ("hybrid+rerank", "Hyb.+rerank"), ("jina", "jina-v3")]
GRID_ARMS = ["F512c", "F256t", "FL", "PP", "S", "S+T", "E", "CR", "E-strong", "CR-strong", "LC-S"]
MECH_ARMS = ["E-noRK", "M", "M-noRK", "M-TP", "RM", "SM", "E+KX", "E+AX", "E+SX", "E(meta+key)", "PF"]
SECONDARY = {"qasper": ["rec@256t", "rec@512t", "rec@1024t", "hit@1", "hit@3", "hit@5", "answer_f1"],
             "freshstack": ["prec@512t", "prec@1024t", "prec@2048t", "rec@1024t", "hit@1", "hit@5", "mrr@10"]}
DS_NAME = {"qasper": "Qasper", "freshstack": "FreshStack"}


def _tex(label: str) -> str:
    return (label.replace("E-strong", "\\Estrong{}").replace("CR-strong", "\\CRstrong{}")
            .replace("-noRK", "$-$K").replace("PP", "P").replace("M-TP", "M$-$TP").replace("_", "\\_"))


def write_grids(out: Path, pm: dict) -> None:
    rows = []
    for f in ("stats.json", "extra/stats_extra.json", "h4_stats.json", "h5_stats.json"):
        d = load(V2 / f)
        rows += (d or {}).get("systems", [])
    mean = {}
    for r in rows:  # first source wins, so the confirmatory files define the shared arms
        mean.setdefault((r["dataset"], r["retriever"], r["system"], r["metric"]), r["mean"])

    def cell(ds, ret, arm, m):
        v = mean.get((ds, ret, arm, m))
        return "--" if v is None else f"{100 * v:.1f}"

    L = ["% Generated by MDKeyChunker-eval/paper_numbers.py; do not edit by hand."]
    for ds in ("qasper", "freshstack"):
        rets = [(k, n) for k, n in GRID_RET if any(key[0] == ds and key[1] == k for key in mean)]
        arms = [a for a in GRID_ARMS if any(key[0] == ds and key[2] == a for key in mean)]
        L += ["\\begin{table*}[!htbp]", "\\centering", "\\footnotesize",
              f"\\begin{{tabular}}{{@{{}}l{'c' * len(rets)}@{{}}}}", "\\toprule",
              "\\textbf{Arm} & " + " & ".join(f"\\textbf{{{n}}}" for _, n in rets) + " \\\\", "\\midrule"]
        for a in arms:
            L.append(_tex(a) + " & " + " & ".join(cell(ds, k, a, pm[ds]) for k, _ in rets) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{DS_NAME[ds]}: mean {pm[ds]} ($\\times$100) for every arm under every retriever. "
              "LC-S: late-chunked S, only under jina-v3; \\CRstrong{} was built for Qasper only. "
              "Per-question values and 95\\% CIs are in the released results.}",
              f"\\label{{tab:grid-{ds}}}", "\\end{table*}", ""]
        mets = SECONDARY[ds]
        arms2 = [a for a in GRID_ARMS + MECH_ARMS if any(key[0] == ds and key[1] == "hybrid" and key[2] == a for key in mean)]
        L += ["\\begin{table*}[!htbp]", "\\centering", "\\footnotesize",
              f"\\begin{{tabular}}{{@{{}}l{'c' * len(mets)}@{{}}}}", "\\toprule",
              "\\textbf{Arm} & " + " & ".join(f"\\textbf{{{_tex(m)}}}" for m in mets) + " \\\\", "\\midrule"]
        for a in arms2:
            L.append(_tex(a) + " & " + " & ".join(cell(ds, "hybrid", a, m) for m in mets) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{DS_NAME[ds]}: secondary metrics ($\\times$100) under the hybrid retriever, "
              "including the mechanism arms (M$-$K: merged E$-$K; E(meta+key): keys and entities also indexed). RM is averaged over five seeds"
              + ("; PF covers 100 papers." if ds == "qasper" else ".") + "}",
              f"\\label{{tab:grid2-{ds}}}", "\\end{table*}", ""]
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
