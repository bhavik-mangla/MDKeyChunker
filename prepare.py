"""Convert a Qasper subset to Markdown documents with question/evidence annotations.

Qasper questions were written by NLP practitioners who saw only title + abstract,
and evidence paragraphs were marked by separate annotators. That removes the
author-written-query circularity of the original MDKeyChunker evaluation.
"""
import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"


def to_markdown(paper: dict) -> str:
    lines = [f"# {paper['title']}", "", "## Abstract", "", paper["abstract"].strip(), ""]
    ft = paper["full_text"]
    for name, paragraphs in zip(ft["section_name"], ft["paragraphs"]):
        name = (name or "").strip()
        if name:
            # Qasper encodes nesting as "Method ::: Encoder ::: Attention"
            parts = [p.strip() for p in name.split(":::")]
            level = min(1 + len(parts), 6)
            lines += [f"{'#' * level} {parts[-1]}", ""]
        for para in paragraphs:
            para = para.strip()
            if para:
                lines += [para, ""]
    return "\n".join(lines)


def text_evidence(answer: dict) -> list[str]:
    # Evidence pointing at tables/figures is not in the running text; drop it.
    return [e for e in answer["evidence"] if e.strip() and not e.startswith("FLOAT SELECTED")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--papers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(DATA / "subset.json"))
    args = ap.parse_args()

    import pyarrow.parquet as pq  # Hugging Face mirror of the official Qasper release

    rows = pq.read_table(DATA / f"qasper-{args.split}.parquet").to_pylist()
    papers = []
    for p in rows:
        qs = []
        qas = p["qas"]
        for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):
            ev: set[str] = set()
            for ans in answers["answer"]:
                if not ans["unanswerable"]:
                    ev.update(text_evidence(ans))
            if ev:
                qs.append({"qid": qid, "question": question, "evidence": sorted(ev)})
        if qs:
            papers.append({"id": p["id"], "title": p["title"], "markdown": to_markdown(p), "questions": qs})

    random.Random(args.seed).shuffle(papers)
    subset = papers[: args.papers]
    Path(args.out).write_text(json.dumps(subset, indent=1))
    nq = sum(len(p["questions"]) for p in subset)
    kb = sum(len(p["markdown"]) for p in subset) / 1024
    print(f"{len(subset)} papers, {nq} questions, {kb:.0f} KB markdown -> {args.out}")


if __name__ == "__main__":
    main()
