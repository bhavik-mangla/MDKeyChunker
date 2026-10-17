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

    # ---------------------------------------------------------------- exploratory v4 analyses (analysis_v4.py)
    ax = load(V2 / "analysis_v4.json")
    if ax:
        write_ax(ax)

    # ---------------------------------------------------------------- spend
    total = 0.0
    led = ROOT / "results" / "spend_ledger.jsonl"
    if led.exists():
        for line in led.read_text().splitlines():
            total += json.loads(line).get("cost_usd") or 0
    put("spend", f"{total:.2f}")

    write_grids(Path(args.out).with_name("grids.tex"), pm)
    write_fig_retrievers(Path(args.out).with_name("fig_retrievers.tex"), pm)
    write_fig_frontier(Path(args.out).with_name("fig_frontier.tex"), pm)
    write_fig_mechanism(Path(args.out).with_name("fig_mechanism.tex"))

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
              "including the add-on arms (M$-$K: merged E$-$K; E(meta+key): keys and entities also indexed). RM is averaged over five seeds"
              + ("; PF covers 100 papers." if ds == "qasper" else ".") + "}",
              f"\\label{{tab:grid2-{ds}}}", "\\end{table*}", ""]
    out.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------- analysis_v4 (exploratory, post hoc)

AX_RET = {"bm25": "bm25", "nomic": "nomic", "mxbai": "mxbai", "hybrid": "hyb", "qwen3": "q3", "hybrid-qwen3": "hq3",
          "hybrid+rerank": "rr", "jina": "jina"}


def axrow(prefix: str, r: dict | None) -> None:
    """row() at 95% for analysis_v4 rows; contrast rows carry no verdict, so derive it from the CI."""
    if r and r.get("status") == "ok" and "verdict" not in r:
        r = {**r, "verdict": "different" if r.get("excludes_zero_95") else "no_difference_detected"}
    row(prefix, r, "95")


def pct(x: float | None, nd: int = 0) -> str:
    return "--" if x is None else f"{100 * x:.{nd}f}"


def write_ax(ax: dict) -> None:
    """Macros with prefix ax- from results/v2/analysis_v4.json. Every number here is exploratory."""
    d = {"qasper": "qasper", "freshstack": "fs"}
    lx = ax.get("lexical", {})
    for ds, k in d.items():
        L = lx.get(ds)
        if not L:
            continue
        for r, t in AX_RET.items():
            if r in L["by_bin"]:
                for b in ("low", "mid", "high"):
                    axrow(f"ax-ovl-{k}-{t}-{b}", L["by_bin"][r][b])
                axrow(f"ax-ovl-{k}-{t}-lmh", L["contrast_low_minus_high"][r])
            B = L["bridging"]["by_retriever"].get(r)
            if B:
                axrow(f"ax-br-{k}-{t}-yes", B["bridged"])
                axrow(f"ax-br-{k}-{t}-miss", B["missing_not_bridged"])
                axrow(f"ax-br-{k}-{t}-none", B["no_missing"])
                axrow(f"ax-br-{k}-{t}-c", B["contrast_bridged_minus_missing_not_bridged"])
                if B.get("share_of_total_gain_bridged") is not None:
                    put(f"ax-br-{k}-{t}-gainshare", pct(B["share_of_total_gain_bridged"]))
            T = L["targeting"].get(r)
            if T:
                for b in ("low", "mid", "high"):
                    axrow(f"ax-tg-{k}-{t}-{b}", T[b])
                axrow(f"ax-tg-{k}-{t}-c", T["contrast_high_minus_low"])
        put(f"ax-ovl-{k}-cut1", pct(L["overlap_cuts"][0]))
        put(f"ax-ovl-{k}-cut2", pct(L["overlap_cuts"][1]))
        put(f"ax-ovl-{k}-mean", pct(L["overlap_mean"]))
        B = L["bridging"]
        put(f"ax-br-{k}-share", pct(B["share_bridged"]))
        put(f"ax-br-{k}-n", str(round(B["share_bridged"] * B["questions_with_gold_chunk"])))
        put(f"ax-br-{k}-nmiss", str(B["n_missing_not_bridged"]))
        put(f"ax-br-{k}-nnone", str(B["n_no_missing"]))
        put(f"ax-br-{k}-missing", f"{B['missing_terms_mean']:.1f}")
        put(f"ax-br-{k}-missbridged", pct(B["share_missing_bridged"]))
        for o, (sh, n) in B["bridged_share_by_outcome_bm25"].items():
            put(f"ax-br-{k}-bm25-{o}share", pct(sh))
            put(f"ax-br-{k}-bm25-{o}n", str(n))
        T = L["targeting"]
        put(f"ax-tg-{k}-gold", f"{T['meta_hits_gold_mean']:.1f}")
        put(f"ax-tg-{k}-other", f"{T['meta_hits_other_mean']:.1f}")
        for arm, v in lx["added_tokens"][ds].items():
            put(f"ax-add-{k}-{arm}-new", f"{v['new_types_mean']:.1f}")
            put(f"ax-add-{k}-{arm}-newc", f"{v['new_content_types_mean']:.1f}")
            put(f"ax-add-{k}-{arm}-text", f"{v['text_types_mean']:.0f}")

    fr = ax.get("frontier", {})
    for ds, k in d.items():
        F = fr.get(ds)
        if not F:
            continue
        for name, r in F["comparisons"].items():
            if r.get("status") == "ok":
                axrow(f"ax-fr-{k}-{name.replace('_minus_', '-').replace('_', '')}", r)
        for arm, v in fr["index_tokens"][ds].items():
            put(f"ax-tok-{k}-{arm}", f"{v['tokens_per_chunk']:,.0f}")
        for tag in ("best", "best_free", "best_free_no_rerank"):
            g = F[tag]
            t_ = tag.replace("_no_rerank", "-norr").replace("_", "")
            put(f"ax-fr-{k}-{t_}", f"{g['arm']} + {g['retriever']}".replace("hybrid+rerank", "rerank"))
            put(f"ax-fr-{k}-{t_}-m", pt(g["score"], sign=False))

    he = ax.get("heterogeneity", {})
    for ds, k in d.items():
        for key, v in he.get("win_tie_loss", {}).get(ds, {}).items():
            pair, r = key.split("|")
            if pair == "E-S+T" and r in AX_RET:
                for o in ("win", "tie", "loss"):
                    put(f"ax-wtl-{k}-{AX_RET[r]}-{o}", pct(v[o], 1))
    Q = he.get("qasper")
    if Q:
        for g, groups in Q["groups"].items():
            for lab, v in groups.items():
                gl = lab.replace("/", "").replace("+", "p")
                gg = g.replace("_", "")
                put(f"ax-het-qasper-{gg}-{gl}-n", str(v["n"]))
                for r in ("hybrid", "bm25", "hybrid+rerank"):
                    axrow(f"ax-het-qasper-{gg}-{gl}-{AX_RET[r]}", v[r])
        for r in ("hybrid", "bm25"):
            axrow(f"ax-het-qasper-ev4v1-{AX_RET[r]}", Q["contrast_evidence_4plus_minus_1"][r])
            axrow(f"ax-het-qasper-lenhl-{AX_RET[r]}", Q["contrast_paper_chunks_high_minus_low"][r])
        f1 = Q["answer_f1"]
        for lab in ("extractive", "abstractive", "mixed", "yes/no"):
            axrow(f"ax-f1-{lab.replace('/', '')}", f1[lab]["E-S+T"])
        for lab, v in f1["conditional_on_recall_change"].items():
            axrow(f"ax-f1-{lab.replace('_', '')}", v["dF1"])
            put(f"ax-f1-{lab.replace('_', '')}-n", str(v["n"]))
        axrow("ax-f1-updown", f1["contrast_rec_up_minus_rec_down"])
        put("ax-f1-recchanged", pct(f1["share_rec_changed"]))
        put("ax-f1-rho", f"{f1['spearman_drec_dF1']:.2f}")
    Fh = he.get("freshstack")
    if Fh:
        for lab, v in Fh["groups"]["gold_length"].items():
            for r in ("hybrid", "bm25", "hybrid+rerank"):
                axrow(f"ax-het-fs-gold-{lab}-{AX_RET[r]}", v[r])
        for dom, v in Fh["groups"]["domain"].items():
            for r in ("bm25", "hybrid+rerank", "qwen3"):
                axrow(f"ax-het-fs-{dom}-{AX_RET[r]}", v[r])
        for r in ("hybrid", "bm25"):
            axrow(f"ax-het-fs-goldhl-{AX_RET[r]}", Fh["contrast_gold_long_minus_short"][r])

    bu = ax.get("budgets", {})
    for ds, k in d.items():
        Bd = bu.get(ds)
        if not Bd:
            continue
        for key, r in Bd["rows"].items():
            pair, ret, m = key.split("|")
            axrow(f"ax-bud-{k}-{pair.replace('+', 'p')}-{AX_RET[ret]}-{m}", r)
        put(f"ax-bud-{k}-nrev", str(len(Bd["sign_reversals"])))
        put(f"ax-bud-{k}-nrevstrict", str(sum(1 for x in Bd["sign_reversals"] if x.get("strict"))))

    lc = ax.get("late_chunking")
    if lc:
        for name in ("loss_vs_dcos", "loss_vs_cos_late", "loss_vs_n_chunks", "loss_vs_tokens", "dcos_vs_n_chunks"):
            v = lc[f"rho_{name}"]
            nm = name.replace("_vs_", "-").replace("_", "")
            put(f"ax-lc-rho-{nm}", f"{v['rho']:.2f}".replace("-", "$-$"))
            put(f"ax-lc-rho-{nm}-ci", "[" + ", ".join(f"{x:.2f}".replace("-", "$-$") for x in v["ci95_pct"]) + "]")
        for lab, tag in (("1", "one"), (">1", "multi")):
            axrow(f"ax-lc-win-{tag}", lc["by_windows"][lab])
            put(f"ax-lc-win-{tag}-n", str(lc["by_windows"][lab]["papers"]))
        axrow("ax-lc-win-c", lc["by_windows"]["contrast_multi_minus_one"])
        axrow("ax-lc-dcos-c", lc["by_dcos_tertile"]["contrast_high_minus_low"])
        put("ax-lc-share-ge90", pct(lc["share_papers_cos_late_ge_0_9"], 1))

    gr = ax.get("granularity", {})
    for ds, k in d.items():
        G = gr.get(ds)
        if not G:
            continue
        for fam in ("text", "meta"):
            put(f"ax-gran-{k}-{fam}-rho", f"{G[fam]['spearman_len_score_hybrid']:.2f}".replace("-", "$-$"))
            put(f"ax-gran-{k}-{fam}-n", str(len(G[fam]["points"])))
            for p_ in G[fam]["points"]:
                put(f"ax-gran-{k}-{p_['arm']}-len", f"{p_['mean_tokens']:.0f}")
        for pair, v in G["merge_by_gold_length"].items():
            if isinstance(v, dict) and "contrast_long_minus_short" in v:
                tag = pair.replace("-E-noRK", "").replace("-E", "")
                for lab in ("low", "mid", "high"):
                    axrow(f"ax-gran-{k}-{tag}-{lab}", v[lab])
                axrow(f"ax-gran-{k}-{tag}-c", v["contrast_long_minus_short"])

    ky = ax.get("keys", {})
    for ds, k in d.items():
        for arm, v in ky.get(ds, {}).items():
            a = arm.replace("-noRK", "nork")
            put(f"ax-key-{k}-{a}-distmed", f"{v['reuse_distance_median']:.0f}")
            put(f"ax-key-{k}-{a}-adj", pct(v["share_reuse_adjacent"]))
            put(f"ax-key-{k}-{a}-within3", pct(v["share_reuse_within_3"]))
            put(f"ax-key-{k}-{a}-gt10", pct(v["share_reuse_gt_10"]))
            put(f"ax-key-{k}-{a}-evictdocs", str(v["docs_with_eviction"]))
            put(f"ax-key-{k}-{a}-outside", str(v["reuses_of_key_outside_window"]))
            put(f"ax-key-{k}-{a}-reuses", f"{v['reuses']:,}")
            put(f"ax-key-{k}-{a}-verbatim", pct(v["key_verbatim_in_text"]))
            put(f"ax-key-{k}-{a}-keytokintext", pct(v["key_token_share_in_text"]))
            put(f"ax-key-{k}-{a}-kwnew", pct(v["keyword_token_share_not_in_st"]))
            put(f"ax-key-{k}-{a}-qnew", pct(v["question_token_share_not_in_st"]))


# ---------------------------------------------------------------------------- figure: E - S+T by retriever

FIG_RET = [("bm25", "BM25"), ("hybrid", "Hybrid (primary)"), ("hybrid-qwen3", "BM25+qwen3"),
           ("nomic", "nomic"), ("mxbai", "mxbai"), ("jina", "jina-v3"), ("qwen3", "qwen3"), ("hybrid+rerank", "Hybrid+rerank")]
FIG_SPLIT = 3  # rows above the divider rank with BM25


def write_fig_retrievers(out: Path, pm: dict) -> None:
    """Dot-and-whisker plot of E - S+T (points, 95% BCa CI) per retriever and dataset, as TikZ."""
    s, x = load(V2 / "stats.json") or {}, load(V2 / "extra" / "stats_extra.json") or {}
    rows = s.get("primary", []) + s.get("exploratory", []) + x.get("rows", [])
    def get(ds, ret):
        for r in rows:
            if (r.get("dataset") == ds and r.get("retriever") == ret and r.get("metric") == pm[ds]
                    and r.get("a") == "E" and r.get("b") == "S+T" and r.get("status", "ok") == "ok"):
                return 100 * r["diff"], 100 * r["ci95_bca"][0], 100 * r["ci95_bca"][1]
        return None
    xs, x0, x1, dy = 0.42, -4, 11, 0.62  # cm per point, axis range, row height
    X = lambda v: f"{(v - x0) * xs:.3f}"
    n = len(FIG_RET)
    L = ["% Generated by MDKeyChunker-eval/paper_numbers.py; do not edit by hand.",
         "\\begin{tikzpicture}[font=\\scriptsize]"]
    top = n * dy
    L.append(f"\\draw[gray!60, dashed] ({X(0)},0) -- ({X(0)},{top:.3f});")
    for t in range(-4, 12, 2):
        lab = "0" if t == 0 else (f"+{t}" if t > 0 else f"$-${-t}")
        L.append(f"\\draw ({X(t)},0) -- ({X(t)},-0.08) node[below] {{{lab}}};")
    L.append(f"\\draw ({X(x0)},0) -- ({X(x1)},0);")
    ys = top - FIG_SPLIT * dy
    L.append(f"\\draw[gray!50] ({X(x0)},{ys:.3f}) -- ({X(x1)},{ys:.3f});")
    L.append(f"\\node at ({X((x0 + x1) / 2)},-0.55) {{E $-$ S+T (points; 95\\% CI)}};")
    for i, (ret, name) in enumerate(FIG_RET):
        y = top - (i + 0.5) * dy
        L.append(f"\\node[anchor=east] at ({X(x0)},{y:.3f}) {{{name}}};")
        for ds, off, mark in (("qasper", 0.1, "fill=black, circle, inner sep=1.3pt"),
                              ("freshstack", -0.1, "draw=black, fill=white, rectangle, inner sep=1.3pt")):
            v = get(ds, ret)
            if v is None:
                continue
            d, lo, hi = v
            yy = y + off
            L.append(f"\\draw ({X(lo)},{yy:.3f}) -- ({X(hi)},{yy:.3f});")
            L.append(f"\\node[{mark}] at ({X(d)},{yy:.3f}) {{}};")
    ly = top + 0.25
    L.append(f"\\node[fill=black, circle, inner sep=1.3pt] at ({X(5.2)},{ly:.3f}) {{}};"
             f"\\node[anchor=west] at ({X(5.5)},{ly:.3f}) {{Qasper}};")
    L.append(f"\\node[draw=black, fill=white, rectangle, inner sep=1.3pt] at ({X(8.6)},{ly:.3f}) {{}};"
             f"\\node[anchor=west] at ({X(8.9)},{ly:.3f}) {{FreshStack}};")
    L.append("\\end{tikzpicture}")
    out.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------- figure: indexing spend vs retriever

FIG_ARMS = [("S+T", "free"), ("E", "E"), ("CR", "CR"), ("E-strong", "E$_{s}$"), ("CR-strong", "CR$_{s}$")]
FIG_LINES = [("bm25", "BM25", "densely dotted"), ("hybrid", "Hybrid", "solid"),
             ("qwen3", "qwen3", "dashed"), ("hybrid+rerank", "Hybrid+rerank", "very thick")]


def write_fig_frontier(out: Path, pm: dict) -> None:
    """Primary-metric mean per indexing arm (x) and retriever (lines), one panel per dataset, as TikZ."""
    rows = []
    for f in ("stats.json", "extra/stats_extra.json"):
        rows += (load(V2 / f) or {}).get("systems", [])
    mean = {}
    for r in rows:
        mean.setdefault((r["dataset"], r["retriever"], r["system"], r["metric"]), 100 * r["mean"])
    L = ["% Generated by MDKeyChunker-eval/paper_numbers.py; do not edit by hand.",
         "\\begin{tikzpicture}[font=\\scriptsize]"]
    W, H, gap = 4.6, 3.6, 1.2
    for k, (ds, title) in enumerate((("qasper", "Qasper (rec@512t)"), ("freshstack", "FreshStack (prec@1024t)"))):
        ox = k * (W + gap)
        arms = [(a, lab) for a, lab in FIG_ARMS if any((ds, r, a, pm[ds]) in mean for r, _, _ in FIG_LINES)]
        vals = [mean[(ds, r, a, pm[ds])] for a, _ in arms for r, _, _ in FIG_LINES if (ds, r, a, pm[ds]) in mean]
        lo, hi = 5 * int(min(vals) // 5), 5 * int(max(vals) // 5 + 1)
        Y = lambda v: (v - lo) / (hi - lo) * H
        Xa = lambda i: ox + (i + 0.5) * W / len(arms)
        L.append(f"\\draw ({ox:.2f},0) rectangle ({ox + W:.2f},{H:.2f});")
        L.append(f"\\node[above] at ({ox + W / 2:.2f},{H:.2f}) {{{title}}};")
        for t in range(lo, hi + 1, 5):
            L.append(f"\\draw[gray!25] ({ox:.2f},{Y(t):.3f}) -- ({ox + W:.2f},{Y(t):.3f});"
                     f"\\node[left] at ({ox:.2f},{Y(t):.3f}) {{{t}}};")
        tok = ((load(V2 / "analysis_v4.json") or {}).get("frontier", {}).get("index_tokens", {}).get(ds, {}))
        for i, (a, lab) in enumerate(arms):
            t = tok.get(a, {}).get("in_per_chunk", 0) if a != "S+T" else 0
            tl = "0" if not t else f"{t / 1000:.1f}k"
            L.append(f"\\node[below, align=center] at ({Xa(i):.3f},0) {{{lab}\\\\[-1pt]{{\\color{{gray}}{tl}}}}};")
        for r, name, style in FIG_LINES:
            pts = [(Xa(i), Y(mean[(ds, r, a, pm[ds])])) for i, (a, _) in enumerate(arms) if (ds, r, a, pm[ds]) in mean]
            if len(pts) > 1:
                L.append(f"\\draw[{style}] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in pts) + ";")
            for x, y in pts:
                L.append(f"\\fill ({x:.3f},{y:.3f}) circle (1.1pt);")
    ly = -1.0
    for j, (r, name, style) in enumerate(FIG_LINES):
        x = 0.2 + j * 2.6
        L.append(f"\\draw[{style}] ({x:.2f},{ly}) -- ({x + 0.6:.2f},{ly});\\node[anchor=west] at ({x + 0.65:.2f},{ly}) {{{name}}};")
    L.append("\\end{tikzpicture}")
    out.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------- figure: document-expansion mechanism

FIG_MECH_RET = [("bm25", "BM25"), ("hybrid", "Hybrid"), ("hybrid-qwen3", "BM25+qwen3"),
                ("nomic", "nomic"), ("qwen3", "qwen3"), ("hybrid+rerank", "Hyb.+rerank")]
FIG_MECH_GRP = [("bridged", "missing term supplied", "black"),
                ("missing_not_bridged", "missing term not supplied", "white"),
                ("no_missing", "no missing term", "gray!45")]


def write_fig_mechanism(out: Path) -> None:
    """Qasper E - S+T (points, 95% CI) by question group and retriever, as TikZ bars."""
    b = ((load(V2 / "analysis_v4.json") or {}).get("lexical", {}).get("qasper", {})
         .get("bridging", {}).get("by_retriever", {}))
    y0, y1, H, bw, gw = -8, 32, 3.4, 0.3, 1.5
    Y = lambda v: (v - y0) / (y1 - y0) * H
    L = ["% Generated by MDKeyChunker-eval/paper_numbers.py; do not edit by hand.",
         "\\begin{tikzpicture}[font=\\scriptsize]"]
    W = len(FIG_MECH_RET) * gw
    for t in range(y0, y1 + 1, 8):
        L.append(f"\\draw[gray!25] (0,{Y(t):.3f}) -- ({W:.2f},{Y(t):.3f});"
                 f"\\node[left] at (0,{Y(t):.3f}) {{{'0' if t == 0 else (f'+{t}' if t > 0 else f'$-${-t}')}}};")
    L.append(f"\\draw (0,{Y(0):.3f}) -- ({W:.2f},{Y(0):.3f});")
    for i, (r, name) in enumerate(FIG_MECH_RET):
        cx = (i + 0.5) * gw
        L.append(f"\\node[below] at ({cx:.2f},0) {{{name}}};")
        for j, (g, _, fill) in enumerate(FIG_MECH_GRP):
            row = b.get(r, {}).get(g)
            if not row or "diff" not in row:
                continue
            d, lo, hi = (100 * row["diff"], 100 * row["ci95_bca"][0], 100 * row["ci95_bca"][1])
            x = cx + (j - 1) * bw
            L.append(f"\\filldraw[fill={fill}, draw=black, line width=0.3pt] ({x - bw / 2:.3f},{Y(0):.3f}) rectangle ({x + bw / 2:.3f},{Y(d):.3f});")
            L.append(f"\\draw[line width=0.4pt] ({x:.3f},{Y(max(lo, y0)):.3f}) -- ({x:.3f},{Y(min(hi, y1)):.3f});")
    L.append(f"\\draw (0,0) rectangle ({W:.2f},{H:.2f});")
    L.append(f"\\node[rotate=90] at (-0.75,{H / 2:.2f}) {{E $-$ S+T, Qasper (points)}};")
    for j, (g, lab, fill) in enumerate(FIG_MECH_GRP):
        x = (0.0, 3.2, 6.9)[j]
        L.append(f"\\filldraw[fill={fill}, draw=black] ({x:.2f},{H + 0.18:.2f}) rectangle ({x + 0.18:.2f},{H + 0.36:.2f});"
                 f"\\node[anchor=west] at ({x + 0.22:.2f},{H + 0.27:.2f}) {{{lab}}};")
    L.append("\\end{tikzpicture}")
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
