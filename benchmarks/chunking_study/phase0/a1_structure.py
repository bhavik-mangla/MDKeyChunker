"""Analysis 1 (Qasper, v3 PILOT data): is the RQ1 structure gain an evidence-unit alignment artefact?

(a) greedy-oracle rec@512t per chunk set (the ceiling each chunking allows)
(b) share of evidence paragraphs split across >= 2 windows
(c) structure-blind paragraph packer (Stage-1 size, headers ignored) and a
    length-matched token window; BM25 / nomic / hybrid rec@512t
(d) evidence scored as max over annotators vs union (all systems, all comparisons)
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (ENC, chunks_of, covered, eval_qasper, load_qasper, mean_of, ngrams,  # noqa: E402
                    render, retrieved_units, save_json, unit_grams, words, ROOT, OUT)
from stats_pilot import analyse  # noqa: E402

MAX_CHARS, MIN_CHARS = 1500, 100     # mdkeychunker Config defaults used by Stage 1 in v3
RETR = ["bm25", "nomic", "hybrid"]


# ------------------------------------------------------------ new chunkers
def para_pack(md: str) -> list[dict]:
    """Pack blank-line-separated blocks up to MAX_CHARS, never splitting a block.
    Header lines are ordinary blocks: they neither start nor end a chunk.
    Small chunks merge forward (cap 2x max) exactly as Stage 1 does."""
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]
    chunks, cur, size = [], [], 0
    for b in blocks:
        if cur and size + len(b) > MAX_CHARS:
            chunks.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
    if cur:
        chunks.append("\n\n".join(cur))
    merged = []
    for c in chunks:
        if merged and len(merged[-1]) < MIN_CHARS and len(merged[-1]) + len(c) <= 2 * MAX_CHARS:
            merged[-1] += "\n\n" + c
        else:
            merged.append(c)
    if len(merged) >= 2 and len(merged[-1]) < MIN_CHARS:
        last = merged.pop()
        merged[-1] += "\n\n" + last
    return [{"text": t} for t in merged]


def fixed_tokens(md: str, size: int, overlap: int) -> list[dict]:
    toks = ENC.encode(md)
    step = size - overlap
    return [{"text": ENC.decode(toks[i:i + size])} for i in range(0, max(len(toks) - overlap, 1), step)]


# ------------------------------------------------------------ (a) oracle
def _progress(ev_list, units, grams):
    s = 0.0
    for e in ev_list:
        ws = words(e)
        if len(ws) < 5:
            s += float(covered(e, units, grams))
        else:
            eg = ngrams(ws)
            s += min(1.0, (sum(g in grams for g in eg) / len(eg)) / 0.5)
    return s


def oracle_rec(chunks: list[dict], evidence: list[str], budget: int = 512) -> float:
    toks = [ENC.encode(c["text"]) for c in chunks]
    best = 0.0
    for per_token in (False, True):
        chosen, used, units = [], 0, []
        while used < budget:
            grams = unit_grams(units)
            base = _progress(evidence, units, grams)
            cand = None
            for i, t in enumerate(toks):
                if i in chosen:
                    continue
                tt = t[: budget - used]
                u = words(ENC.decode(tt))
                gain = _progress(evidence, units + [u], grams | ngrams(u)) - base
                if gain <= 1e-9:
                    continue
                score = gain / len(tt) if per_token else gain
                if cand is None or score > cand[0]:
                    cand = (score, i, len(tt), u)
            if cand is None:
                break
            chosen.append(cand[1])
            used += cand[2]
            units.append(cand[3])
        g = unit_grams(units)
        best = max(best, sum(covered(e, units, g) for e in evidence) / len(evidence))
    return best


# ------------------------------------------------------------ (b) splits
def split_stats(chunks: list[dict], evidence: list[str]):
    cw = [words(c["text"]) for c in chunks]
    cg = [ngrams(w) for w in cw]
    rows = []
    for e in evidence:
        ws = words(e)
        if len(ws) < 5:
            continue
        eg = ngrams(ws)
        best = max(sum(g in G for g in eg) / len(eg) for G in cg)
        rows.append(best)
    return rows


# ------------------------------------------------------------ (d) annotator sets
def annotator_sets() -> dict:
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "data" / "qasper-dev.parquet").to_pylist()
    out = {}
    for p in rows:
        qas = p["qas"]
        for qid, answers in zip(qas["question_id"], qas["answers"]):
            sets = []
            for ans in answers["answer"]:
                if ans["unanswerable"]:
                    continue
                ev = [e for e in ans["evidence"] if e.strip() and not e.startswith("FLOAT SELECTED")
                      and ":::" not in e]
                if ev:
                    sets.append(sorted(set(ev)))
            out[qid] = sets
    return out


def main():
    papers = load_qasper()
    subset = {p["id"]: p for p in json.loads((ROOT / "data" / "subset.json").read_text())}
    struct_tok = [len(ENC.encode(c["text"])) for p in papers for c in chunks_of("struct", p["id"])]
    L = int(statistics.median(struct_tok))
    ov = round(L / 8)  # same 1/8 overlap ratio as fixedtok (32 of 256)
    print(f"median struct chunk = {L} cl100k tokens; length-matched window {L} tok, overlap {ov}")

    built = {}
    for p in papers:
        md = subset[p["id"]]["markdown"]
        built[("parapack", p["id"])] = para_pack(md)
        built[("fixedlen", p["id"])] = fixed_tokens(md, L, ov)

    def get(cs):
        return (lambda pid: built[(cs, pid)]) if cs in ("parapack", "fixedlen") else (lambda pid: chunks_of(cs, pid))

    sets = ["fixed512", "fixedtok", "fixedlen", "parapack", "struct", "merged_rk", "merged_nork"]
    stats = {}
    for cs in sets:
        n = [len(get(cs)(p["id"])) for p in papers]
        t = [len(ENC.encode(c["text"])) for p in papers for c in get(cs)(p["id"])]
        stats[cs] = {"chunks": sum(n), "median_tokens": statistics.median(t), "mean_tokens": sum(t) / len(t)}

    # (a) oracle
    oracle = {}
    for cs in sets:
        per = {}
        for p in papers:
            ch = get(cs)(p["id"])
            per[p["id"]] = [oracle_rec(ch, q["evidence"]) for q in p["questions"]]
        oracle[cs] = {"mean": mean_of(per), "per": per}
        print(f"oracle rec@512t {cs:12} {oracle[cs]['mean']:.3f}")

    # (b) split share over unique (paper, evidence >= 5 words)
    split = {}
    for cs in sets:
        best = []
        for p in papers:
            ev = sorted({e for q in p["questions"] for e in q["evidence"]})
            best += split_stats(get(cs)(p["id"]), ev)
        split[cs] = {"n_evidence": len(best),
                     "share_not_whole_in_one_chunk": sum(b < 1.0 for b in best) / len(best),
                     "share_below_50pct_in_any_chunk": sum(b < 0.5 for b in best) / len(best)}
        print(f"split {cs:12} {split[cs]}")

    # (c) + (d): rankings for all systems; score union and max-over-annotators in one pass
    ann = annotator_sets()
    for p in papers:
        for q in p["questions"]:
            q["ann"] = ann[q["qid"]]

    def ev_fn(q, order, chunks):
        units = retrieved_units(order, chunks, budget=512)
        g = unit_grams(units)
        cov = {e: covered(e, units, g) for e in set(q["evidence"]) | {e for s in q["ann"] for e in s}}
        union = sum(cov[e] for e in q["evidence"]) / len(q["evidence"])
        mx = max(sum(cov[e] for e in s) / len(s) for s in q["ann"])
        return {"rec@512t": union, "max@512t": mx}

    systems = [("fixed512", "text"), ("fixedtok", "text"), ("fixedlen", "text"), ("parapack", "text"),
               ("struct", "text"), ("struct", "tc"), ("cr", "cr"), ("enr_rk", "meta"), ("enr_nork", "meta"),
               ("merged_rk", "text"), ("merged_rk", "meta"), ("merged_nork", "meta")]
    per = {}
    for cs, mode in systems:
        res = eval_qasper(papers, get(cs), lambda pid, ch, mode=mode: [render(c, mode) for c in ch],
                          RETR + ["mxbai"], metrics=("rec@512t", "max@512t"), evidence_fn=ev_fn)
        for (r, m), v in res.items():
            per[(r, f"{cs}/{mode}", m)] = v

    n_multi = sum(len({tuple(s) for s in q["ann"]}) >= 2 for p in papers for q in p["questions"])
    nq = sum(len(p["questions"]) for p in papers)

    means = {"|".join(k): mean_of(v) for k, v in per.items()}
    comps_c = [("parapack/text", "fixedtok/text"), ("parapack/text", "fixed512/text"),
               ("struct/text", "parapack/text"), ("struct/text", "fixedlen/text"),
               ("fixedlen/text", "fixedtok/text"), ("struct/text", "fixedtok/text")]
    part_c = []
    for a, b in comps_c:
        for r in RETR:
            part_c.append({"a": a, "b": b, "retriever": r,
                           **analyse(per[(r, a, "rec@512t")], per[(r, b, "rec@512t")], 0.05)})

    prereg = [("struct/text", "fixedtok/text"), ("struct/text", "fixed512/text"), ("enr_rk/meta", "struct/tc"),
              ("enr_rk/meta", "cr/cr"), ("merged_rk/meta", "merged_nork/meta"), ("merged_rk/meta", "enr_rk/meta")]
    part_d = []
    for a, b in prereg:
        for r in RETR + ["mxbai"]:
            row = {"a": a, "b": b, "retriever": r}
            for m in ("rec@512t", "max@512t"):
                row[m] = analyse(per[(r, a, m)], per[(r, b, m)], 0.05)
            part_d.append(row)

    save_json("a1_structure.json", {
        "label": "PILOT (v3 data, 30 papers / 79 questions); exploratory",
        "length_matched_window_tokens": L, "overlap": ov, "chunk_stats": stats,
        "oracle_rec512t": {k: v["mean"] for k, v in oracle.items()},
        "oracle_per_question": {k: v["per"] for k, v in oracle.items()},
        "split": split, "means": means, "part_c": part_c, "part_d": part_d,
        "questions_with_>=2_distinct_annotator_sets": n_multi, "questions": nq,
    })
    # per-question dump for re-analysis
    save_json("a1_per_question.json", {"|".join(k): v for k, v in per.items()})

    lines = ["# A1 Qasper structure confound (PILOT v3 data, exploratory)", "",
             f"Length-matched window: {L} cl100k tokens (median struct chunk), overlap {ov}.", "",
             "| chunk set | chunks | median tok | greedy-oracle rec@512t | evidence not whole in 1 chunk | evidence <50% in any chunk |",
             "|---|---|---|---|---|---|"]
    for cs in sets:
        lines.append(f"| {cs} | {stats[cs]['chunks']} | {stats[cs]['median_tokens']:.0f} | {oracle[cs]['mean']:.3f} | "
                     f"{100*split[cs]['share_not_whole_in_one_chunk']:.1f}% | {100*split[cs]['share_below_50pct_in_any_chunk']:.1f}% |")
    lines += ["", "rec@512t means (union evidence):", "",
              "| system | " + " | ".join(RETR + ["mxbai"]) + " |", "|---" * 5 + "|"]
    for cs, mode in systems:
        s = f"{cs}/{mode}"
        lines.append(f"| {s} | " + " | ".join(f"{means[f'{r}|{s}|rec@512t']:.3f}" for r in RETR + ["mxbai"]) + " |")
    lines += ["", "(c) paired differences rec@512t (points):", "", "| A - B | retr | diff | 95% CI | p_perm | TOST±5 p |",
              "|---|---|---|---|---|---|"]
    for c in part_c:
        lines.append(f"| {c['a']} - {c['b']} | {c['retriever']} | {100*c['diff']:+.1f} | "
                     f"[{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] | {c['p_perm']:.3f} | {c['p_tost']:.3f} |")
    lines += ["", f"(d) union vs max-over-annotators ({n_multi}/{nq} questions have >=2 distinct annotator evidence sets):", "",
              "| A - B | retr | union diff [95% CI] | max diff [95% CI] |", "|---|---|---|---|"]
    for row in part_d:
        u, m = row["rec@512t"], row["max@512t"]
        lines.append(f"| {row['a']} - {row['b']} | {row['retriever']} | {100*u['diff']:+.1f} [{100*u['ci95'][0]:+.1f}, {100*u['ci95'][1]:+.1f}] | "
                     f"{100*m['diff']:+.1f} [{100*m['ci95'][0]:+.1f}, {100*m['ci95'][1]:+.1f}] |")
    (OUT / "a1_structure.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
