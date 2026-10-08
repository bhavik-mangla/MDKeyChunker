"""Analysis 1c addendum: fixed-window results depend strongly on where boundaries fall.

The single length-matched window (203 tok, overlap 25) scored far below the
256/32 window on BM25 (0.199 vs 0.265). This script measures that boundary
sensitivity and builds a fairer length-matched baseline: the per-question
average over 4 window phases (start offsets 0, L/4, L/2, 3L/4).
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ENC, ROOT, OUT, chunks_of, eval_qasper, load_qasper, mean_of, save_json  # noqa: E402
from stats_pilot import analyse  # noqa: E402

RETR = ["bm25", "nomic", "hybrid"]


def windows(md: str, size: int, overlap: int, offset: int = 0) -> list[dict]:
    toks = ENC.encode(md)
    out = [{"text": ENC.decode(toks[:offset])}] if offset else []
    step = size - overlap
    for i in range(offset, max(len(toks) - overlap, offset + 1), step):
        out.append({"text": ENC.decode(toks[i:i + size])})
    return out


def main():
    papers = load_qasper()
    md = {p["id"]: p["markdown"] for p in json.loads((ROOT / "data" / "subset.json").read_text())}
    L = int(statistics.median(len(ENC.encode(c["text"])) for p in papers for c in chunks_of("struct", p["id"])))
    ov = round(L / 8)
    text = lambda pid, ch: [c["text"] for c in ch]  # noqa: E731

    # (i) BM25 sensitivity to window size / overlap / phase
    sens = []
    for size in (128, 180, 203, 230, 256, 300):
        o = round(size / 8)
        for off_frac in (0, 0.25, 0.5, 0.75):
            off = int(size * off_frac)
            per = eval_qasper(papers, lambda pid: windows(md[pid], size, o, off), text, ["bm25"])
            sens.append({"size": size, "overlap": o, "offset": off, "bm25_rec@512t": mean_of(per[("bm25", "rec@512t")])})
            print(sens[-1])

    # (ii) phase-averaged length-matched baseline vs struct / fixedtok / parapack
    phases = [int(L * f) for f in (0, 0.25, 0.5, 0.75)]
    avg = {}
    for off in phases:
        per = eval_qasper(papers, lambda pid: windows(md[pid], L, ov, off), text, RETR)
        for k, v in per.items():
            acc = avg.setdefault(k, {pid: [0.0] * len(x) for pid, x in v.items()})
            for pid, x in v.items():
                acc[pid] = [a + b / len(phases) for a, b in zip(acc[pid], x)]
    ref = json.loads((OUT / "a1_per_question.json").read_text())
    rows = []
    for r in RETR:
        a = avg[(r, "rec@512t")]
        row = {"retriever": r, "fixedlen_phaseavg": mean_of(a)}
        for b in ("struct/text", "fixedtok/text", "parapack/text"):
            res = analyse(ref[f"{r}|{b}|rec@512t"], a, 0.05)
            row[f"{b} - fixedlen_phaseavg"] = res
        rows.append(row)
        print(r, row["fixedlen_phaseavg"],
              {k: (round(100 * v["diff"], 1), [round(100 * x, 1) for x in v["ci95"]]) for k, v in row.items() if isinstance(v, dict)})
    # (iii) paragraph packer at smaller max sizes (size-matched to struct): is it headers or size?
    import a1_structure as A
    pp_rows = []
    for mx in (600, 800, 1000, 1500):
        A.MAX_CHARS = mx
        built = {pid: A.para_pack(md[pid]) for pid in md}
        med = statistics.median(len(ENC.encode(c["text"])) for p in papers for c in built[p["id"]])
        per = eval_qasper(papers, lambda pid: built[pid], text, RETR)
        row = {"max_chars": mx, "median_tokens": med, "chunks": sum(len(built[p["id"]]) for p in papers)}
        for r in RETR:
            row[r] = mean_of(per[(r, "rec@512t")])
            row[f"struct - parapack ({r})"] = analyse(ref[f"{r}|struct/text|rec@512t"], per[(r, "rec@512t")], 0.05)
        pp_rows.append(row)
        print(mx, med, {r: round(row[r], 3) for r in RETR},
              {r: (round(100 * row[f"struct - parapack ({r})"]["diff"], 1),
                   [round(100 * x, 1) for x in row[f"struct - parapack ({r})"]["ci95"]]) for r in RETR})
    save_json("a1c_window_phase.json", {"label": "PILOT v3 data, exploratory", "L": L, "overlap": ov,
                                        "bm25_sensitivity": sens, "phase_avg": rows, "parapack_sizes": pp_rows})


if __name__ == "__main__":
    main()
