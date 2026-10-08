"""Analysis 2 (Qasper, PILOT v3 data): BM25 with corpus-level IDF and a k1/b grid.

IDF and avgdl are fitted over the rendered chunks of ALL 30 papers for the
system being scored (so generated text still shapes its own statistics, as in
any real index); scoring/ranking stays within each paper, as in v3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, chunks_of, eval_qasper, load_qasper, mean_of, rank, render, save_json, tok  # noqa: E402
from stats_pilot import analyse  # noqa: E402

SYSTEMS = {"S": ("struct", "text"), "S+T": ("struct", "tc"), "E": ("enr_rk", "meta"), "CR": ("cr", "cr")}
COMPS = [("E", "S+T"), ("E", "S"), ("E", "CR"), ("CR", "S+T"), ("S+T", "S")]


def main():
    papers = load_qasper()
    grid = [(k1, b) for k1 in (0.9, 1.2, 1.5) for b in (0.4, 0.75)]
    per = {}
    for name, (cs, mode) in SYSTEMS.items():
        docs = {p["id"]: [render(c, mode) for c in chunks_of(cs, p["id"])] for p in papers}
        corpus = [tok(d) for pid in docs for d in docs[pid]]
        for idf in ("paper", "corpus"):
            for k1, b in grid:
                stats = corpus if idf == "corpus" else None

                def rf(pid, r, d, qs, k1=k1, b=b, stats=stats):
                    return rank(r, d, qs, bm25_kw={"k1": k1, "b": b}, stats_corpus=stats)
                retr = ["bm25"] + (["hybrid"] if (k1, b) == (1.5, 0.75) else [])
                res = eval_qasper(papers, lambda pid, cs=cs: chunks_of(cs, pid), lambda pid, ch, docs=docs: docs[pid],
                                  retr, rank_fn=rf)
                for (r, m), v in res.items():
                    per[(r, idf, k1, b, name)] = v
    rows = []
    for (r, idf, k1, b, name), v in per.items():
        if name != "E":
            continue
        row = {"retriever": r, "idf": idf, "k1": k1, "b": b,
               "means": {n: mean_of(per[(r, idf, k1, b, n)]) for n in SYSTEMS}}
        for a, bb in COMPS:
            row[f"{a} - {bb}"] = analyse(per[(r, idf, k1, b, a)], per[(r, idf, k1, b, bb)], 0.05)
        rows.append(row)
    save_json("a2_bm25.json", {"label": "PILOT v3 data, exploratory", "rows": rows})
    lines = ["# A2 BM25 IDF scope and k1/b (Qasper rec@512t, PILOT)", "",
             "| retr | IDF | k1 | b | S | S+T | E | CR | E-S+T [95% CI] | E-S [95% CI] | E-CR [95% CI] |", "|---" * 11 + "|"]
    for r in rows:
        m = r["means"]
        c = lambda k: f"{100*r[k]['diff']:+.1f} [{100*r[k]['ci95'][0]:+.1f}, {100*r[k]['ci95'][1]:+.1f}]"  # noqa: E731
        lines.append(f"| {r['retriever']} | {r['idf']} | {r['k1']} | {r['b']} | {m['S']:.3f} | {m['S+T']:.3f} | {m['E']:.3f} | "
                     f"{m['CR']:.3f} | {c('E - S+T')} | {c('E - S')} | {c('E - CR')} |")
    (OUT / "a2_bm25.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
