"""Corpus-level retrieval evaluation (FreshStack Laravel docs).

Unlike Qasper, each question searches the pooled chunks of every document.
Gold evidence is large verbatim Markdown spans, so relevance is judged per
retrieved chunk: a chunk is relevant when at least half of its word 5-grams
occur in the question's gold spans. Budgeted metrics measure how much of a
fixed token budget, filled in rank order, is gold text (precision) and how
much of the gold text it covers (recall).
"""
import argparse
import json
from pathlib import Path

from evaluate import (CACHE, SYSTEMS, ngrams, print_comparisons, print_table, rank, render,
                      retrieved_units, summarize, unit_grams, words)

ROOT = Path(__file__).parent


def doc_id(path: str) -> str:
    return path.replace("/", "__")


def relevant(chunk_text: str, gold_grams: set) -> bool:
    grams = ngrams(words(chunk_text))
    return bool(grams) and sum(g in gold_grams for g in grams) / len(grams) >= 0.5


def question_metrics(order: list[int], chunks: list[dict], gold_grams: set) -> dict:
    rel = [relevant(chunks[i]["text"], gold_grams) for i in order[:10]]
    m = {f"hit@{k}": float(any(rel[:k])) for k in (1, 3, 5, 10)}
    m["mrr@10"] = next((1.0 / (r + 1) for r, x in enumerate(rel) if x), 0.0)
    for budget in (512, 1024, 2048):
        got = unit_grams(retrieved_units(order, chunks, budget=budget))  # per-chunk grams
        inter = len(got & gold_grams)
        m[f"prec@{budget}t"] = inter / len(got) if got else 0.0
        m[f"rec@{budget}t"] = inter / len(gold_grams) if gold_grams else 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "freshstack_laravel.json"))
    ap.add_argument("--retrievers", default="bm25,mxbai,nomic,hybrid")
    ap.add_argument("--out", default=str(ROOT / "results" / "freshstack_results.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    ids = [doc_id(d["path"]) for d in data["docs"]]
    systems = [s for s in SYSTEMS if all((CACHE / s[0] / f"{i}.json").exists() for i in ids)]

    per: dict = {}
    qs = data["questions"]
    QIDS = {q["qid"]: [q["qid"]] for q in qs}
    for cs, mode in systems:
        chunks = [c for i in ids for c in json.loads((CACHE / cs / f"{i}.json").read_text())["chunks"]]
        docs = [render(c, mode) for c in chunks]
        for r in args.retrievers.split(","):
            orders = rank(r, docs, [q["question"] for q in qs])
            for q, order in zip(qs, orders):
                gold = set().union(*(ngrams(words(e["text"])) for e in q["evidence"]))
                for metric, v in question_metrics(order, chunks, gold).items():
                    # Questions are independent here; cluster = question
                    per.setdefault((r, f"{cs}/{mode}", metric), {})[q["qid"]] = [v]

    results = summarize(per)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    # Per-question values for re-analysis: "retriever|system|metric" -> {cluster: [values in question order]}
    per_q = {"|".join(k): v for k, v in per.items()}
    Path(args.out.replace(".json", "_per_question.json")).write_text(json.dumps(
        {"question_ids": QIDS, "values": per_q}))
    print_comparisons(results, "prec@1024t")
    for metric in ("prec@1024t", "rec@1024t", "hit@5", "mrr@10"):
        print_table(results, metric)


if __name__ == "__main__":
    main()
