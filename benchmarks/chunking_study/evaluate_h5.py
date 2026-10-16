"""Amendment 3 (H5, H6, sec. C). Retrieval and metrics exactly as evaluate_v2.

H5  related keys at retrieval time, on the cached E chunks (hybrid ranking of the meta render):
    E+KX  after each retrieved chunk, insert its key-linked chunks from the same source document
          (same key, key in its related keys, or its key in their related keys), in document order
    E+AX  count-matched control: the same number of chunks per retrieved chunk, taken from its
          nearest neighbours in the same document (next, previous, next+1, ...)
    Holm over 4 two-sided sign-flip tests (H5a E+KX - E, H5b E+KX - E+AX; both datasets).
H6  E - PF on the 100-paper Qasper sample (TOST +/-5, single test).
C   exploratory: E rendered with key and entities ("meta+key") - E, all retrievers.

Writes results/v2/h5_{qasper,freshstack}_per_question.json, h5_stats.json, h5_summary.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_v2
import evaluate
import evaluate_corpus
import evaluate_h4
import evaluate_v2 as ev

OUT = ev.OUT
H5 = [("E+KX", "E"), ("E+KX", "E+AX")]


def render_key(c: dict) -> str:
    extra = [c.get("key", ""), ", ".join(e.get("name", "") for e in c.get("entities", []) if isinstance(e, dict))]
    return evaluate.render(c, "meta") + "\n" + "\n".join(x for x in extra if x)


def links(chunks: list[dict], doc_of: list[str]) -> list[list[int]]:
    by_doc: dict[str, list[int]] = {}
    for i, d in enumerate(doc_of):
        by_doc.setdefault(d, []).append(i)
    out: list[list[int]] = [[] for _ in chunks]
    for idx in by_doc.values():
        for i in idx:
            ki, ri = chunks[i].get("key", ""), set(chunks[i].get("related_keys", []))
            for j in idx:
                if j == i:
                    continue
                kj, rj = chunks[j].get("key", ""), set(chunks[j].get("related_keys", []))
                if (ki and (ki == kj or ki in rj)) or (kj and kj in ri):
                    out[i].append(j)
    return out


def section_expand(order: list[int], chunks: list[dict], doc_of: list[str]) -> list[int]:
    """Amendment 4 B: parent-section expansion (same document and section path), not count-matched."""
    out, inc = [], set()
    for r in order:
        if r in inc:
            continue
        out.append(r)
        inc.add(r)
        sec = chunks[r].get("section", "")
        new = [j for j in range(len(chunks)) if j not in inc and doc_of[j] == doc_of[r]
               and chunks[j].get("section", "") == sec] if sec else []
        out += new
        inc.update(new)
    return out


def expand(order: list[int], lk: list[list[int]], doc_of: list[str]) -> tuple[list[int], list[int]]:
    """Returns (KX order, AX order)."""
    kx, inc, added = [], set(), {}
    for r in order:
        if r in inc:
            continue
        kx.append(r)
        inc.add(r)
        new = [j for j in sorted(lk[r]) if j not in inc]
        kx += new
        inc.update(new)
        added[r] = len(new)
    ax, inc = [], set()
    n = len(doc_of)
    for r in order:
        if r not in inc:
            ax.append(r)
            inc.add(r)
        want, step = added.get(r, 0), 1
        while want > 0 and (r + step < n or r - step >= 0):
            for j in (r + step, r - step):
                if want > 0 and 0 <= j < n and doc_of[j] == doc_of[r] and j not in inc:
                    ax.append(j)
                    inc.add(j)
                    want -= 1
            step += 1
            if all(not (0 <= j < n and doc_of[j] == doc_of[r]) for j in (r + step, r - step)):
                break
    return kx, ax


def in_budget(order: list[int], chunks: list[dict], budget: int) -> list[int]:
    """Chunk indices that contribute text within the token budget (as evaluate.retrieved_units)."""
    out, used = [], 0
    for i in order:
        if used >= budget:
            break
        out.append(i)
        used += len(evaluate.ENC.encode(chunks[i]["text"]))
    return out


def _systems(chunks, doc_of, queries, retrievers, rank_fn=evaluate.rank):
    """Yields (label, retriever, per-question orders, chunks used for scoring)."""
    meta = ev.rank_all([evaluate.render(c, "meta") for c in chunks], queries, retrievers, rank_fn)
    keyed = ev.rank_all([render_key(c) for c in chunks], queries, retrievers, rank_fn)
    lk = links(chunks, doc_of)
    for r in retrievers:
        exp = [expand(o, lk, doc_of) for o in meta[r]]
        yield "E", r, meta[r]
        yield "E+KX", r, [k for k, _ in exp]
        yield "E+AX", r, [a for _, a in exp]
        yield "E(meta+key)", r, keyed[r]
        yield "E+SX", r, [section_expand(o, chunks, doc_of) for o in meta[r]]


def run_eval(retrievers: list[str]) -> None:
    sel = json.loads((ev.V2 / "selection.json").read_text())
    cache = build_v2.CACHE_V2
    load = lambda variant, d: json.loads(build_v2.cache_path("local", variant, d, cache).read_text())["chunks"]  # noqa: E731

    papers = {p["id"]: p for p in json.loads((ev.V2 / "qasper_v2.json").read_text())}
    per, qids, links_stat, retrieved_stat = {}, {}, [0, 0], [0, 0]
    pf_ids = {p.stem for p in (cache / "local" / "perfield").glob("*.json")}
    for pid in sel["qasper"]["paper_ids"]:
        qs = [q for q in papers[pid]["questions"] if q["evidence"]]
        if not qs:
            continue
        qids[pid] = [q["qid"] for q in qs]
        chunks = load("enr", pid)
        doc_of = [pid] * len(chunks)
        lk = links(chunks, doc_of)
        links_stat[0] += sum(1 for x in lk if x)
        links_stat[1] += len(lk)
        for label, r, orders in _systems(chunks, doc_of, [q["question"] for q in qs], retrievers):
            for q, order in zip(qs, orders):
                for m, v in evaluate.question_metrics(order, chunks, q["evidence"]).items():
                    per.setdefault((r, label, m), {}).setdefault(pid, []).append(v)
                if label == "E" and r == "hybrid":
                    got = in_budget(order, chunks, 512)
                    retrieved_stat[0] += sum(1 for i in got if lk[i])
                    retrieved_stat[1] += len(got)
        if pid in pf_ids:
            pf = load("perfield", pid)
            orders = ev.rank_all([evaluate.render(c, "meta") for c in pf], [q["question"] for q in qs], retrievers)
            for r in retrievers:
                for q, order in zip(qs, orders[r]):
                    for m, v in evaluate.question_metrics(order, pf, q["evidence"]).items():
                        per.setdefault((r, "PF", m), {}).setdefault(pid, []).append(v)
    ev.dump_per_question(OUT / "h5_qasper_per_question.json", {"per": per, "question_ids": qids},
                         {"chunks_with_links": links_stat, "retrieved_with_links": retrieved_stat,
                          "pf_papers": sorted(pf_ids)})
    print(f"qasper: {len(qids)} papers, chunks with >=1 key link {links_stat[0]}/{links_stat[1]}, "
          f"PF papers {len(pf_ids)}", flush=True)

    per, strata, links_stat, retrieved_stat = {}, {}, [0, 0], [0, 0]
    for dom in ev.FS_DOMAINS:
        keep = set(sel["freshstack"][dom]["qids"])
        qs = [q for q in json.loads((ev.V2 / f"freshstack_{dom}.json").read_text())["questions"] if q["qid"] in keep]
        chunks, doc_of = [], []
        for d in sel["freshstack"][dom]["doc_ids"]:
            c = load("enr", d)
            chunks += c
            doc_of += [d] * len(c)
        lk = links(chunks, doc_of)
        links_stat[0] += sum(1 for x in lk if x)
        links_stat[1] += len(lk)
        golds = [set().union(*(evaluate.ngrams(evaluate.words(e["text"])) for e in q["evidence"])) for q in qs]
        for q in qs:
            strata[q["qid"]] = dom
        for label, r, orders in _systems(chunks, doc_of, [q["question"] for q in qs], retrievers):
            for q, gold, order in zip(qs, golds, orders):
                for m, v in evaluate_corpus.question_metrics(order, chunks, gold).items():
                    per.setdefault((r, label, m), {})[q["qid"]] = [v]
                if label == "E" and r == "hybrid":
                    got = in_budget(order, chunks, 1024)
                    retrieved_stat[0] += sum(1 for i in got if lk[i])
                    retrieved_stat[1] += len(got)
    ev.dump_per_question(OUT / "h5_freshstack_per_question.json",
                         {"per": per, "question_ids": {k: [k] for k in strata}, "strata": strata},
                         {"chunks_with_links": links_stat, "retrieved_with_links": retrieved_stat})
    print(f"freshstack: {len(strata)} questions, chunks with >=1 key link {links_stat[0]}/{links_stat[1]}", flush=True)


def analyze(n: int = 10_000) -> dict:
    data = {ds: ev.load_per_question(OUT / f"h5_{ds}_per_question.json") for ds in ev.DATASETS
            if (OUT / f"h5_{ds}_per_question.json").exists()}
    specs = [ev._spec("h5", ds, "hybrid", ev.PRIMARY_METRIC[ds], a, b, None, "p_perm")
             for ds in ev.DATASETS for a, b in H5]
    h5 = ev.run_family(specs, data, n)
    h6 = ev.run_family([ev._spec("h6", "qasper", "hybrid", "rec@512t", "E", "PF", ev.MARGIN["qasper"], "p_tost")],
                       data, n)
    expl = []
    for ds, res in data.items():
        per = res["per"]
        for r in ev.RETRIEVERS:
            m = ev.PRIMARY_METRIC[ds]
            for a, b in H5 + [("E(meta+key)", "E"), ("E", "PF"), ("E+SX", "E"), ("E+KX", "E+SX"), ("E+AX", "E")]:
                if (r, a, m) in per and (r, b, m) in per and not (r == "hybrid" and (a, b) in H5 + [("E", "PF")]):
                    expl.append({"family": "exploratory", "dataset": ds, "retriever": r, "metric": m, "a": a, "b": b,
                                 **ev.compare(per[(r, a, m)], per[(r, b, m)], strata=res.get("strata"), n=n)})
        if res.get("strata"):
            m = ev.PRIMARY_METRIC[ds]
            for dom in sorted(set(res["strata"].values())):
                for a, b in H5:
                    sa = {q: v for q, v in per[("hybrid", a, m)].items() if res["strata"][q] == dom}
                    sb = {q: v for q, v in per[("hybrid", b, m)].items() if res["strata"][q] == dom}
                    expl.append({"family": "exploratory", "dataset": f"{ds}/{dom}", "retriever": "hybrid",
                                 "metric": m, "a": a, "b": b, **ev.compare(sa, sb, n=n)})
    links = {ds: {k: json.loads((OUT / f"h5_{ds}_per_question.json").read_text()).get(k)
                  for k in ("chunks_with_links", "retrieved_with_links")} for ds in data}
    result = {"h5": h5, "h6": h6, "exploratory": expl, "systems": ev.system_means(data, n),
              "pf_cost": pf_cost(), "key_links": links}
    (OUT / "h5_stats.json").write_text(json.dumps(result, indent=1))
    L = ["Amendment 3 results (H5 related keys, H6 per-field). A - B, BCa CIs.", "",
         "H5 family (hybrid; Holm over 4 two-sided sign-flip tests)"]
    L += [evaluate_h4._row(r) for r in h5]
    L += ["  key links [linked, total]: " + json.dumps(result["key_links"])]
    L += ["", "H6 (Qasper, 100-paper sample; TOST +/-0.05)"] + [evaluate_h4._row(r) for r in h6]
    L += ["", "Per-field cost", json.dumps(result["pf_cost"], indent=1), "", "System means (hybrid, primary metric)"]
    L += [f"  {r['dataset']:<10} {r['system']:<12} {r['mean']:.4f} [{r['ci95_pct'][0]:.3f},{r['ci95_pct'][1]:.3f}]"
          for r in result["systems"] if r["retriever"] == "hybrid" and r["metric"] == ev.PRIMARY_METRIC[r["dataset"]]]
    L += ["", "Exploratory (unadjusted)"] + [evaluate_h4._row(r) for r in expl]
    (OUT / "h5_summary.txt").write_text("\n".join(L) + "\n")
    return result


def pf_cost() -> dict:
    out = {}
    pf = list((build_v2.CACHE_V2 / "local" / "perfield").glob("*.json"))
    ids = {p.stem for p in pf}
    for variant in ("enr", "perfield"):
        # wall seconds are not comparable (E ran beside CR under different concurrency); tokens only
        tot = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "chunks": 0}
        for d in ids:
            j = json.loads(build_v2.cache_path("local", variant, d).read_text())
            for k in ("calls", "in_tokens", "out_tokens"):
                tot[k] += j["stats"].get(k, 0) or 0
            tot["chunks"] += len(j["chunks"])
        out[variant] = tot
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", default=",".join(ev.RETRIEVERS))
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    evaluate.OLLAMA = build_v2.OLLAMA_URL
    if not args.stats_only:
        run_eval(args.retrievers.split(","))
    analyze(args.boot)
    print((OUT / "h5_summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
