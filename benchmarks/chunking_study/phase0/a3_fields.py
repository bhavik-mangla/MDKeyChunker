"""Analysis 3 (PILOT v3 data): which generated field drives the enrichment effect?

Re-renders the existing enr_rk outputs (no new LLM calls) with one field group
prepended to the chunk text, in the same layout as render(c, "meta"):
  q   questions only (doc2query-like)
  ts  title + summary
  kw  keywords
  all title + summary + keywords + questions (= E, enr_rk/meta)
Compared against S (struct/text, no prefix) and E.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (OUT, chunks_of, eval_freshstack, eval_qasper, fs_chunks, load_freshstack,  # noqa: E402
                    load_qasper, mean_of, render, save_json)
from stats_pilot import analyse  # noqa: E402

RETR = ["bm25", "nomic", "hybrid", "mxbai"]


def render_field(c: dict, field: str) -> str:
    if field in ("text", "tc", "meta", "cr"):
        return render(c, field)
    head = {"q": [" ".join(c.get("questions", []))],
            "ts": [c.get("title", ""), c.get("summary", "")],
            "kw": [", ".join(c.get("keywords", []))]}[field]
    return "\n".join(h for h in head if h) + "\n\n" + c["text"]


VARIANTS = {"S": ("struct", "text"), "S+T": ("struct", "tc"), "S+q": ("enr_rk", "q"),
            "S+ts": ("enr_rk", "ts"), "S+kw": ("enr_rk", "kw"), "E": ("enr_rk", "meta")}


def main():
    out = {"label": "PILOT v3 data, exploratory"}
    lines = ["# A3 Field ablation on existing enr_rk outputs (PILOT)", ""]
    for ds in ("qasper", "freshstack"):
        per = {}
        if ds == "qasper":
            papers = load_qasper()
            metric, margin = "rec@512t", 0.05
            for v, (cs, f) in VARIANTS.items():
                res = eval_qasper(papers, lambda pid, cs=cs: chunks_of(cs, pid),
                                  lambda pid, ch, f=f: [render_field(c, f) for c in ch], RETR)
                for (r, m), x in res.items():
                    per[(r, v)] = x
        else:
            data, ids = load_freshstack()
            metric, margin = "prec@1024t", 0.03
            for v, (cs, f) in VARIANTS.items():
                ch = fs_chunks(cs, ids)
                res = eval_freshstack(data, ch, [render_field(c, f) for c in ch], RETR)
                for (r, m), x in res.items():
                    per[(r, v)] = x
        rows = []
        for r in RETR:
            row = {"retriever": r, "means": {v: mean_of(per[(r, v)]) for v in VARIANTS}}
            for v in ("S+q", "S+ts", "S+kw", "E"):
                row[f"{v} - S"] = analyse(per[(r, v)], per[(r, "S")], margin)
                row[f"{v} - S+T"] = analyse(per[(r, v)], per[(r, "S+T")], margin)
            rows.append(row)
        out[ds] = {"metric": metric, "rows": rows}
        lines += [f"## {ds} ({metric})", "", "| retr | " + " | ".join(VARIANTS) + " | q-S | ts-S | kw-S | E-S |", "|---" * 11 + "|"]
        for row in rows:
            c = lambda k: f"{100*row[k]['diff']:+.1f} [{100*row[k]['ci95'][0]:+.1f},{100*row[k]['ci95'][1]:+.1f}]"  # noqa: E731
            lines.append(f"| {row['retriever']} | " + " | ".join(f"{row['means'][v]:.3f}" for v in VARIANTS) +
                         f" | {c('S+q - S')} | {c('S+ts - S')} | {c('S+kw - S')} | {c('E - S')} |")
        lines.append("")
    save_json("a3_fields.json", out)
    (OUT / "a3_fields.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
