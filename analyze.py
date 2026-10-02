"""Turn caches and evaluation results into the paper's numbers and LaTeX tables.

Reports: chunk-set statistics, the rolling-key ablation (does the key
dictionary increase key reuse and merging?), measured LLM cost, and the
retrieval grid with CIs.
"""
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache"


def load(variant: str, ids: list[str]) -> dict:
    out = {}
    for i in ids:
        p = CACHE / variant / f"{i}.json"
        if p.exists():
            out[i] = json.loads(p.read_text())
    return out


def chunk_stats(ids: list[str], variants: list[str]) -> list[dict]:
    rows = []
    for v in variants:
        data = load(v, ids)
        lens = [len(c["text"]) for d in data.values() for c in d["chunks"]]
        if lens:
            rows.append({"variant": v, "docs": len(data), "chunks": len(lens),
                         "median_chars": st.median(lens), "mean_chars": round(st.mean(lens))})
    return rows


def key_stats(ids: list[str], variant: str, merged: str) -> dict:
    """Key reuse within each document: what the rolling dictionary is supposed to raise."""
    enr, mer = load(variant, ids), load(merged, ids)
    n = uniq = shared_chunks = adjacent_same = adjacent_pairs = 0
    related = related_nonempty = 0
    for d in enr.values():
        keys = [c["key"] for c in d["chunks"]]
        cnt = Counter(k for k in keys if k)
        n += len(keys)
        uniq += len(cnt)
        shared_chunks += sum(c for c in cnt.values() if c > 1)
        adjacent_pairs += max(len(keys) - 1, 0)
        adjacent_same += sum(1 for a, b in zip(keys, keys[1:]) if a and a == b)
        related += sum(len(c["related_keys"]) for c in d["chunks"])
        related_nonempty += sum(1 for c in d["chunks"] if c["related_keys"])
    after = sum(len(d["chunks"]) for d in mer.values())
    return {"variant": variant, "chunks": n, "unique_keys": uniq,
            "key_reuse_rate": round(1 - uniq / n, 3) if n else None,
            "chunks_sharing_a_key": shared_chunks,
            "adjacent_same_key": round(adjacent_same / adjacent_pairs, 3) if adjacent_pairs else None,
            "chunks_after_merge": after, "removed_by_merge": n - after,
            "related_keys_per_chunk": round(related / n, 2) if n else None,
            "chunks_with_related_keys": round(related_nonempty / n, 3) if n else None}


def cost_stats(ids: list[str], variant: str) -> dict:
    data = load(variant, ids)
    s = [d["stats"] for d in data.values() if d.get("stats")]
    calls = sum(x.get("calls", 0) for x in s)
    if not calls:
        return {"variant": variant}
    secs = sum(x.get("seconds", 0) for x in s)
    tin = sum(x.get("in_tokens", x.get("in_tokens_evaluated", 0)) for x in s)
    tout = sum(x.get("out_tokens", 0) for x in s)
    return {"variant": variant, "model": s[0].get("model"), "calls": calls,
            "sec_per_call_wall": round(secs / calls, 1),
            "in_tokens_per_call": round(tin / calls), "out_tokens_per_call": round(tout / calls)}


def latex_grid(results: list[dict], metric: str, caption: str, label: str) -> str:
    rows = [r for r in results if r["metric"] == metric]
    rets = sorted({r["retriever"] for r in rows})
    syss = list(dict.fromkeys(r["system"] for r in rows))
    out = ["\\begin{table*}[t]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{l" + "c" * len(rets) + "}", "\\toprule",
           "System & " + " & ".join(rets) + " \\\\", "\\midrule"]
    for s in syss:
        cells = []
        for r in rets:
            x = next(v for v in rows if v["system"] == s and v["retriever"] == r)
            sig = (s != "struct/text") and (x["diff_ci"][0] > 0 or x["diff_ci"][1] < 0)
            cell = f"{x['mean']:.3f} {{\\scriptsize[{x['ci'][0]:.2f},{x['ci'][1]:.2f}]}}"
            cells.append(f"\\textbf{{{cell}}}" if sig else cell)
        out.append(s.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qasper", "freshstack"], default="qasper")
    args = ap.parse_args()
    if args.dataset == "qasper":
        ids = [p["id"] for p in json.loads((ROOT / "data" / "subset.json").read_text())]
        res_path, variants = ROOT / "results" / "qasper.json", \
            ["fixed512", "fixedtok", "struct", "enr_rk", "merged_rk", "merged_nork"]
        ablation = [("enr_rk", "merged_rk"), ("enr_nork", "merged_nork")]
    else:
        ids = [d["id"] for d in json.loads((ROOT / "data" / "freshstack_docs.json").read_text())]
        res_path, variants = ROOT / "results" / "freshstack.json", \
            ["fixed512", "fixedtok", "struct", "enr_rk", "merged_rk"]
        ablation = [("enr_rk", "merged_rk")]

    report = {"chunk_stats": chunk_stats(ids, variants),
              "key_stats": [key_stats(ids, e, m) for e, m in ablation],
              "cost": [cost_stats(ids, v) for v in ("enr_rk", "enr_nork", "cr")]}
    out = ROOT / "results" / f"{args.dataset}_analysis.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if res_path.exists():
        results = json.loads(res_path.read_text())
        metric = "rec@512t" if args.dataset == "qasper" else "prec@1024t"
        tex = latex_grid(results, metric, f"{args.dataset} {metric} (bold: 95\\% CI of difference vs. struct/text excludes 0)",
                         f"tab:{args.dataset}")
        (ROOT / "results" / f"{args.dataset}_table.tex").write_text(tex)
        print(f"\nwrote results/{args.dataset}_table.tex")


if __name__ == "__main__":
    main()
