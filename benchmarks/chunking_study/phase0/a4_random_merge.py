"""Analysis 4 (PILOT v3 data): RQ4b random-merge control.

For each document, make as many merges as key-based restructuring removed
(len(enr_rk) - len(merged_rk)), but choose what to merge at random within the
document, starting from the same enr_rk chunks. Two controls, 5 seeds each:
  rand_any  any two current chunks of the document
  rand_adj  two currently adjacent chunks
The merged-chunk size cap (3,000 chars, Restructurer's max_merged_size) is kept.
Merged chunks follow Restructurer: text joined in document order, title/summary
from the first member, keyword/question lists unioned (order-preserving).
"""
import random
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (OUT, chunks_of, eval_freshstack, eval_qasper, load_freshstack, load_qasper,  # noqa: E402
                    mean_of, render, save_json)
from evaluate_corpus import doc_id  # noqa: E402,F401
from stats_pilot import analyse  # noqa: E402

RETR = ["bm25", "nomic", "hybrid"]
CAP = 3000
SEEDS = range(5)


def merge_group(chunks, idxs):
    idxs = sorted(idxs)
    g = [chunks[i] for i in idxs]
    u = lambda f: list(dict.fromkeys(x for c in g for x in c.get(f, [])))  # noqa: E731
    return {"text": "\n\n".join(c["text"] for c in g), "section": g[0].get("section", ""),
            "title": g[0].get("title", ""), "summary": g[0].get("summary", ""),
            "keywords": u("keywords"), "questions": u("questions")}


def random_merge(chunks, n_merges, seed, adjacent):
    rng = random.Random(seed)
    groups = [[i] for i in range(len(chunks))]
    size = lambda g: sum(len(chunks[i]["text"]) for i in g) + 2 * (len(g) - 1)  # noqa: E731
    done = 0
    for _ in range(n_merges):
        for _try in range(200):
            if len(groups) < 2:
                break
            if adjacent:
                a = rng.randrange(len(groups) - 1)
                b = a + 1
            else:
                a, b = sorted(rng.sample(range(len(groups)), 2))
            if size(groups[a]) + size(groups[b]) + 2 <= CAP:
                groups[a] = groups[a] + groups[b]
                del groups[b]
                done += 1
                break
    groups.sort(key=min)
    return [merge_group(chunks, g) for g in groups], done


def main():
    out = {"label": "PILOT v3 data, exploratory"}
    lines = ["# A4 RQ4b random-merge control (PILOT)", ""]
    for ds in ("qasper", "freshstack"):
        if ds == "qasper":
            papers = load_qasper()
            doc_ids = [p["id"] for p in papers]
            metric, margin = "rec@512t", 0.05
        else:
            data, doc_ids = load_freshstack()
            metric, margin = "prec@1024t", 0.03
        enr = {d: chunks_of("enr_rk", d) for d in doc_ids}
        n_rm = {d: len(enr[d]) - len(chunks_of("merged_rk", d)) for d in doc_ids}

        def evaluate(get):
            if ds == "qasper":
                return eval_qasper(papers, get, lambda pid, ch: [render(c, "meta") for c in ch], RETR)
            ch = [c for d in doc_ids for c in get(d)]
            return eval_freshstack(data, ch, [render(c, "meta") for c in ch], RETR)

        ref = {"merged_rk": evaluate(lambda d: chunks_of("merged_rk", d)), "enr_rk": evaluate(lambda d: enr[d])}
        res, achieved = {}, {}
        for ctrl in ("rand_any", "rand_adj"):
            seeds = []
            for s in SEEDS:
                built = {}
                tot = 0
                for d in doc_ids:
                    built[d], k = random_merge(enr[d], n_rm[d], s * 1000 + zlib.crc32(d.encode()) % 997, ctrl == "rand_adj")
                    tot += k
                achieved[(ctrl, s)] = tot
                seeds.append(evaluate(lambda d: built[d]))
            res[ctrl] = seeds
        rows = []
        for r in RETR:
            key = (r, metric)
            for ctrl in ("rand_any", "rand_adj"):
                per_seed = [mean_of(sd[key]) for sd in res[ctrl]]
                avg = {k: [sum(sd[key][k][i] for sd in res[ctrl]) / len(SEEDS) for i in range(len(v))]
                       for k, v in res[ctrl][0][key].items()}
                rows.append({"retriever": r, "control": ctrl, "seed_means": per_seed,
                             "merged_rk": mean_of(ref["merged_rk"][key]), "enr_rk": mean_of(ref["enr_rk"][key]),
                             "merged_rk - control(seed-avg)": analyse(ref["merged_rk"][key], avg, margin),
                             "control(seed-avg) - enr_rk": analyse(avg, ref["enr_rk"][key], margin)})
        out[ds] = {"metric": metric, "merges_target": sum(n_rm.values()),
                   "merges_achieved": {f"{c}/{s}": v for (c, s), v in achieved.items()}, "rows": rows}
        lines += [f"## {ds} ({metric}); key-based merges removed {sum(n_rm.values())} chunks", "",
                  "| retr | control | seed means | ctrl mean | merged_rk/meta | enr_rk/meta | merged_rk - ctrl [95% CI] | ctrl - enr_rk [95% CI] |",
                  "|---" * 8 + "|"]
        for row in rows:
            a, b = row["merged_rk - control(seed-avg)"], row["control(seed-avg) - enr_rk"]
            lines.append(f"| {row['retriever']} | {row['control']} | {', '.join(f'{x:.3f}' for x in row['seed_means'])} | "
                         f"{sum(row['seed_means'])/len(SEEDS):.3f} | {row['merged_rk']:.3f} | {row['enr_rk']:.3f} | "
                         f"{100*a['diff']:+.1f} [{100*a['ci95'][0]:+.1f}, {100*a['ci95'][1]:+.1f}] | "
                         f"{100*b['diff']:+.1f} [{100*b['ci95'][0]:+.1f}, {100*b['ci95'][1]:+.1f}] |")
        lines.append("")
    save_json("a4_random_merge.json", out)
    (OUT / "a4_random_merge.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
