"""Qasper v2: the confirmatory sample (ANALYSIS_PLAN_v2.md section 1).

Pool = Qasper dev + test papers minus the 30 pilot papers in data/subset.json.
A paper is eligible if at least one question has text evidence under the v3
rules (prepare.py), tightened as specified for v2:
  - evidence starting with "FLOAT SELECTED" (tables/figures) is dropped;
  - evidence that is a section-name path ("A ::: B") is dropped;
  - evidence from annotators who marked the question unanswerable is ignored.
300 papers are drawn with random.Random(2027) after sorting the pool by paper
id, so the sample does not depend on parquet row order.

Output data/v2/qasper_v2.json: list of papers in the v3 schema (id, title,
markdown, questions[qid, question, evidence]) plus "split" per paper and, per
question, "evidence_by_annotator" and "answers" (one entry per annotator, all
annotators kept, including unanswerable ones, as the official Qasper answer-F1
evaluator expects).
"""
import argparse
import collections
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config

from prepare import to_markdown

DATA = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA / "v2"
SEED = 2027
N_PAPERS = 300


def keep_evidence(e: str) -> bool:
    e = e.strip()
    return bool(e) and not e.startswith("FLOAT SELECTED") and ":::" not in e


def answer_record(ans: dict, annotation_id: str, worker_id: str) -> dict:
    """Mirror the official qasper_evaluator answer typing."""
    if ans["unanswerable"]:
        typ, text = "none", "Unanswerable"
    elif ans["extractive_spans"]:
        typ, text = "extractive", ", ".join(ans["extractive_spans"])
    elif ans["free_form_answer"]:
        typ, text = "abstractive", ans["free_form_answer"]
    elif ans["yes_no"] is not None:
        typ, text = "boolean", "Yes" if ans["yes_no"] else "No"
    else:  # official evaluator raises here; never observed, kept explicit
        typ, text = "empty", ""
    return {"annotation_id": annotation_id, "worker_id": worker_id,
            "type": typ, "text": text, "unanswerable": ans["unanswerable"],
            "extractive_spans": list(ans["extractive_spans"] or []),
            "free_form_answer": ans["free_form_answer"] or "",
            "yes_no": ans["yes_no"]}


def paper_record(p: dict, split: str) -> dict | None:
    qas, qs = p["qas"], []
    for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):
        recs, by_ann, union = [], [], set()
        for ans, aid, wid in zip(answers["answer"], answers["annotation_id"], answers["worker_id"]):
            recs.append(answer_record(ans, aid, wid))
            ev = [] if ans["unanswerable"] else sorted({e for e in ans["evidence"] if keep_evidence(e)})
            by_ann.append(ev)
            union.update(ev)
        if union:
            qs.append({"qid": qid, "question": question, "evidence": sorted(union),
                       "evidence_by_annotator": by_ann, "answers": recs})
    if not qs:
        return None
    return {"id": p["id"], "title": p["title"], "split": split,
            "markdown": to_markdown(p), "questions": qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=N_PAPERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT_DIR / "qasper_v2.json"))
    args = ap.parse_args()

    pilot = {p["id"] for p in json.loads((DATA / "subset.json").read_text())}
    pool, seen = [], set()
    n_pilot = n_noev = 0
    for split in ("dev", "test"):
        for p in pq.read_table(DATA / f"qasper-{split}.parquet").to_pylist():
            assert p["id"] not in seen, p["id"]
            seen.add(p["id"])
            if p["id"] in pilot:
                n_pilot += 1
                continue
            rec = paper_record(p, split)
            if rec is None:
                n_noev += 1
                continue
            pool.append(rec)
    assert n_pilot == len(pilot), (n_pilot, len(pilot))
    pool.sort(key=lambda r: r["id"])
    sample = random.Random(args.seed).sample(pool, args.papers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sample, ensure_ascii=False, indent=1))

    # ---- stats ----
    enc = tiktoken.get_encoding("cl100k_base")
    chunker = MarkdownChunker(Config(log_level="WARNING"))
    nq = sum(len(p["questions"]) for p in sample)
    splits = collections.Counter(p["split"] for p in sample)
    ann_types = collections.Counter(a["type"] for p in sample for q in p["questions"] for a in q["answers"])
    first_types = collections.Counter(q["answers"][0]["type"] for p in sample for q in p["questions"])
    n_ann = collections.Counter(len(q["answers"]) for p in sample for q in p["questions"])
    kb = sum(len(p["markdown"].encode()) for p in sample) / 1024
    n_chunks = cr_tokens = doc_tokens = 0
    for p in sample:
        k = len(chunker.chunk(p["markdown"]))
        t = len(enc.encode(p["markdown"]))
        n_chunks += k
        cr_tokens += k * t
        doc_tokens += t
    # evidence sanity: share of evidence paragraphs present verbatim in the Markdown
    ev = [(e, p["markdown"]) for p in sample for q in p["questions"] for e in q["evidence"]]
    found = sum(e.strip() in md for e, md in ev)
    print(f"pool: {len(pool)} eligible papers ({n_pilot} pilot excluded, "
          f"{n_noev} without text evidence)")
    print(f"sample: {len(sample)} papers {dict(splits)}, {nq} questions, {kb:.0f} KB markdown "
          f"-> {args.out}")
    print(f"annotators per question: {dict(sorted(n_ann.items()))}")
    print(f"answer types (all annotator answers): {dict(ann_types)}")
    print(f"answer types (first annotator): {dict(first_types)}")
    print(f"structural chunks: {n_chunks} (mean {n_chunks / len(sample):.1f}/paper); "
          f"doc tokens {doc_tokens}; CR input tokens (sum over chunks of doc tokens) {cr_tokens}")
    print(f"evidence paragraphs found verbatim in markdown: {found}/{len(ev)}")


if __name__ == "__main__":
    main()
