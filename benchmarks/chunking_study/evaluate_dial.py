"""Exploratory (post hoc): random / similarity merges at M-TP's merge counts on the E-noRK base.

Same retrieval and metrics as evaluate_h4. Writes results/v2/dial_{qasper,freshstack}_per_question.json,
dial_stats.json and dial_summary.txt. Pairs: M-TP - SM-TP, M-TP - RM-TP, SM-TP - E-noRK, RM-TP - E-noRK,
M-TP - E-noRK (unadjusted 95% CIs), primary metric plus FreshStack rec@1024t.
"""
from __future__ import annotations

import json

import build_v2
import evaluate
import evaluate_h4 as h4
import evaluate_v2 as ev

SEEDS = [f"RM-TP{s}" for s in range(5)]
LABELS = {"SM-TP": ("local", "smerge_tp", "meta"), **{f"RM-TP{s}": ("local", f"rmerge_tp_s{s}", "meta") for s in range(5)}}
PAIRS = [("M-TP", "SM-TP"), ("M-TP", "RM-TP"), ("SM-TP", "E-noRK"), ("RM-TP", "E-noRK"), ("M-TP", "E-noRK")]
RETR = ["bm25", "nomic", "hybrid"]


def average(per: dict) -> None:
    for r, label, m in list(per):
        if label != "RM-TP0":
            continue
        seeds = [per[(r, s, m)] for s in SEEDS]
        per[(r, "RM-TP", m)] = {k: [sum(x) / len(x) for x in zip(*(s[k] for s in seeds))] for k in seeds[0]}
    for k in [k for k in per if k[1] in SEEDS]:
        del per[k]


def main() -> None:
    h4.register()
    ev.SYSTEMS_V2.update(LABELS)
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    sel = json.loads((ev.V2 / "selection.json").read_text())
    wanted = ["E-noRK", "M-TP", "SM-TP", *SEEDS]
    cache = build_v2.CACHE_V2
    keep = {"E-noRK", "M-TP", *LABELS}

    def plan(groups):
        saved = dict(ev.SYSTEMS_V2)
        try:
            for k in list(ev.SYSTEMS_V2):
                if k not in keep:
                    del ev.SYSTEMS_V2[k]
            p = ev.plan_systems(groups, cache)
        finally:
            ev.SYSTEMS_V2.clear()
            ev.SYSTEMS_V2.update(saved)
        return {**p, "systems": [s for s in p["systems"] if s in wanted]}

    papers = json.loads((ev.V2 / "qasper_v2.json").read_text())
    pq = plan({pid: [pid] for pid in sel["qasper"]["paper_ids"]})
    res_q = ev.eval_qasper(papers, pq, cache, RETR)
    average(res_q["per"])
    ev.dump_per_question(ev.OUT / "dial_qasper_per_question.json", res_q)
    groups = {d: sel["freshstack"][d]["doc_ids"] for d in ev.FS_DOMAINS}
    dqs = {d: [q for q in json.loads((ev.V2 / f"freshstack_{d}.json").read_text())["questions"]
               if q["qid"] in set(sel["freshstack"][d]["qids"])] for d in ev.FS_DOMAINS}
    res_f = ev.eval_freshstack(dqs, plan(groups), cache, RETR)
    average(res_f["per"])
    ev.dump_per_question(ev.OUT / "dial_freshstack_per_question.json", res_f)

    rows = []
    for ds, res, metrics in (("qasper", res_q, ["rec@512t"]), ("freshstack", res_f, ["prec@1024t", "rec@1024t"])):
        for r in RETR:
            for m in metrics:
                for a, b in PAIRS:
                    ka, kb = (r, a, m), (r, b, m)
                    if ka in res["per"] and kb in res["per"]:
                        rows.append({"family": "exploratory", "dataset": ds, "retriever": r, "metric": m, "a": a,
                                     "b": b, **ev.compare(res["per"][ka], res["per"][kb], strata=res.get("strata"))})
    counts = {}
    for v in ("smerge_tp", *[f"rmerge_tp_s{s}" for s in range(5)]):
        tgt = done = 0
        for p in (cache / "local" / v).glob("*.json"):
            st = json.loads(p.read_text())["stats"]
            tgt += st["n_merges_target"]
            done += st["n_merges_done"]
        counts[v] = {"target": tgt, "done": done}
    (ev.OUT / "dial_stats.json").write_text(json.dumps({"label": "EXPLORATORY (post hoc)", "rows": rows,
                                                        "merge_counts": counts}, indent=1))
    L = ["Exploratory (post hoc): controls at M-TP merge counts on the E-noRK base. Unadjusted 95% BCa CIs.",
         json.dumps(counts)]
    L += [h4._row(r) for r in rows]
    (ev.OUT / "dial_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
