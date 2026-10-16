"""Erratum (v3 PILOT, FreshStack-Laravel): compare corrected zero-LLM results with the
published pilot numbers, and scope the pending LLM-system rebuild.

Inputs: results/freshstack.json (+ _per_question) as published, and
results/phase0/erratum_freshstack.json (+ _per_question) from erratum_eval.py.
Optionally results/phase0/erratum_freshstack_provisional.json: LLM chunk sets taken
unchanged from the pilot cache (stale for the changed file) with corrected gold.

Writes results/phase0/erratum_compare.json and data/cache_fixed/REBUILD_LIST.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, ENC, OUT, ROOT, rank, render  # noqa: E402
from evaluate_corpus import doc_id  # noqa: E402

ZERO = ["fixed512/text", "fixedtok/text", "struct/text", "struct/tc"]
LLM = [("enr_rk", "meta"), ("merged_rk", "meta"), ("merged_rk", "text"), ("cr", "cr")]
RETR = ["bm25", "mxbai", "nomic", "hybrid"]


def key(x):
    return (x["kind"], x["retriever"], x.get("system", x.get("a")), x.get("b"), x["metric"])


def load(p):
    return {key(x): x for x in json.loads(Path(p).read_text())}


def scoring_window(order, chunks, k=10, budget=2048):
    """Chunk indices that can influence any metric: top-10 plus the 2048-token budget fill."""
    win, used = set(order[:k]), 0
    for i in order:
        win.add(i)
        used += len(ENC.encode(chunks[i]["text"]))
        if used >= budget:
            break
    return win


def main() -> None:
    pub = load(ROOT / "results" / "freshstack.json")
    new = load(OUT / "erratum_freshstack.json")
    prov_p = OUT / "erratum_freshstack_provisional.json"
    prov = load(prov_p) if prov_p.exists() else {}
    data_rep = json.loads((OUT / "erratum_data.json").read_text())
    affected = set(data_rep["affected_qids"])

    # 1. per-system means, zero-LLM
    sys_rows = []
    for k, x in new.items():
        if k[0] != "system" or x["system"] not in ZERO:
            continue
        p = pub[k]
        sys_rows.append({"retriever": k[1], "system": k[2], "metric": k[4], "published": p["mean"],
                         "corrected": x["mean"], "delta": x["mean"] - p["mean"],
                         "published_diff_vs_struct": p["diff_vs_struct"], "corrected_diff_vs_struct": x["diff_vs_struct"],
                         "sig_vs_struct_published": p["diff_ci"][0] > 0 or p["diff_ci"][1] < 0,
                         "sig_vs_struct_corrected": x["diff_ci"][0] > 0 or x["diff_ci"][1] < 0})

    # 2. pre-registered comparisons (zero-LLM: RQ1; provisional: RQ2-RQ4)
    def comps(src, tag):
        rows = []
        for k, x in src.items():
            if k[0] != "comparison":
                continue
            p = pub[k]
            rows.append({"source": tag, "retriever": k[1], "a": k[2], "b": k[3], "metric": k[4],
                         "published": p["diff"], "published_ci": p["ci"], "published_sig": p["excludes_zero"],
                         "corrected": x["diff"], "corrected_ci": x["ci"], "corrected_sig": x["excludes_zero"],
                         "delta": x["diff"] - p["diff"]})
        return rows
    comp_rows = comps(new, "corrected")
    if prov:
        zero_keys = {(r["retriever"], r["a"], r["b"], r["metric"]) for r in comp_rows}
        comp_rows += [r for r in comps(prov, "provisional")
                      if (r["retriever"], r["a"], r["b"], r["metric"]) not in zero_keys]

    # 3. per-question: which questions' values moved (zero-LLM)
    pq_pub = json.loads((ROOT / "results" / "freshstack_per_question.json").read_text())["values"]
    pq_new = json.loads((OUT / "erratum_freshstack_per_question.json").read_text())["values"]
    moved = {}
    for k, v in pq_new.items():
        for qid, val in v.items():
            if val != pq_pub[k][qid]:
                moved.setdefault(qid, set()).add(k)
    moved_q = sorted(moved)

    # 4. LLM systems: which questions could see a different result after the rebuild
    data = json.loads((ROOT / "data" / "freshstack_laravel_fixed.json").read_text())
    ids = [doc_id(d["path"]) for d in data["docs"]]
    qs = data["questions"]
    rebuild = json.loads((ROOT / "data" / "cache_fixed" / "_erratum_chunks.json").read_text())["llm_rebuild"]
    changed_docs = {e["doc_id"]: e for e in rebuild}
    exposure = {}
    for cs, mode in LLM:
        chunks, owner = [], []
        for i in ids:
            cc = json.loads((CACHE / cs / f"{i}.json").read_text())["chunks"]
            chunks += cc
            owner += [(i, j) for j in range(len(cc))]
        docs = [render(c, mode) for c in chunks]
        for r in RETR:
            for q, order in zip(qs, rank(r, docs, [q["question"] for q in qs])):
                win = scoring_window(order, chunks)
                if any(owner[i][0] in changed_docs for i in win):
                    exposure.setdefault(q["qid"], set()).add(f"{r}|{cs}/{mode}")
    gold_in_changed = {q["qid"] for q in qs if any(doc_id(e["path"]) in changed_docs for e in q["evidence"])}
    exposed = set(exposure) | gold_in_changed | affected

    rebuild_list = {
        "why": "laravel/docs commit erratum (pilot used 1e8496c2; FreshStack corpus = 724c31cc). "
               "Rebuild these docs at the text in data/freshstack_laravel_fixed.json into data/cache_fixed/.",
        "settings": {"model": "qwen2.5:7b", "NUM_CTX": 32768, "seed": 0, "temperature": 0,
                     "build": "build.py build_paper (same code as v3), CACHE -> data/cache_fixed"},
        "docs": [{"doc_id": e["doc_id"], "path": e["path"],
                  "struct_chunks": e["struct_chunks_after"],
                  "struct_chunk_indices_changed": e["struct_chunk_indices_changed"],
                  "rebuild": {
                      "enr_rk": {"llm_calls": e["struct_chunks_after"],
                                 "note": "rolling keys chain chunk to chunk; chunks before the first "
                                         "changed index get identical prompts, so only "
                                         f"{e['struct_chunks_after'] - e['struct_chunk_indices_changed'][0]} "
                                         "of them can change, but the whole doc is rebuilt"},
                      "merged_rk": {"llm_calls": 0, "pilot_chunks": e["merged_rk_chunks_pilot"],
                                    "note": "deterministic restructure of the rebuilt enr_rk"},
                      "cr": {"llm_calls": e["struct_chunks_after"],
                             "note": "every CR prompt contains the full document, so all chunks change"}}}
                 for e in rebuild],
        "total_llm_calls": sum(2 * e["struct_chunks_after"] for e in rebuild),
        "unchanged_docs_copy_from": "data/cache/{enr_rk,merged_rk,cr}/<doc_id>.json",
        "questions_possibly_affected": {
            "gold_span_changed": sorted(affected),
            "gold_in_changed_doc": sorted(gold_in_changed),
            "changed_doc_in_scoring_window_any_llm_system": sorted(exposure),
            "union": sorted(exposed), "union_count": len(exposed), "of": len(qs)},
    }
    (ROOT / "data" / "cache_fixed" / "REBUILD_LIST.json").write_text(json.dumps(rebuild_list, indent=1))

    out = {"systems": sys_rows, "comparisons": comp_rows,
           "zero_llm_questions_with_any_value_moved": moved_q,
           "moved_outside_affected": sorted(set(moved_q) - affected),
           "llm_exposure": {q: sorted(v) for q, v in exposure.items()}}
    (OUT / "erratum_compare.json").write_text(json.dumps(out, indent=1))

    # console summary
    big = max(sys_rows, key=lambda r: abs(r["delta"]))
    print(f"zero-LLM system means: {len(sys_rows)} cells; max |delta| {abs(big['delta']):.4f} "
          f"({big['retriever']} {big['system']} {big['metric']})")
    for m in ("prec@1024t", "rec@1024t", "hit@5", "mrr@10"):
        print(f"\n{m}: published -> corrected (delta)")
        for s in ZERO:
            cells = [r for r in sys_rows if r["system"] == s and r["metric"] == m]
            print(f"  {s:15}" + "".join(f"  {r['retriever']:6} {r['published']:.3f}->{r['corrected']:.3f} ({r['delta']:+.3f})"
                                         for r in sorted(cells, key=lambda r: RETR.index(r["retriever"]))))
    flips = [r for r in comp_rows if r["published_sig"] != r["corrected_sig"]]
    print(f"\ncomparisons: {len(comp_rows)}; max |delta| "
          f"{max(abs(r['delta']) for r in comp_rows):.4f}; significance flips: {len(flips)}")
    for r in flips:
        print(f"  FLIP [{r['source']}] {r['retriever']} {r['a']} - {r['b']} {r['metric']}: "
              f"{r['published']:+.3f} {r['published_ci']} -> {r['corrected']:+.3f} {r['corrected_ci']}")
    for r in comp_rows:
        if r["metric"] == "prec@1024t":
            print(f"  [{r['source'][:4]}] {r['retriever']:6} {r['a']:15} - {r['b']:15} "
                  f"{r['published']:+.3f} -> {r['corrected']:+.3f} ({r['delta']:+.3f}) "
                  f"sig {r['published_sig']}->{r['corrected_sig']}")
    print(f"\nzero-LLM: questions with any per-question value moved: {len(moved_q)} "
          f"({len(set(moved_q) & affected)} of the {len(affected)} affected; "
          f"{len(set(moved_q) - affected)} others)")
    print(f"LLM rebuild: {rebuild_list['total_llm_calls']} LLM calls; questions possibly affected: "
          f"{len(exposed)}/{len(qs)} (gold changed {len(affected)}, gold in changed doc {len(gold_in_changed)}, "
          f"changed doc in scoring window {len(exposure)})")


if __name__ == "__main__":
    main()
