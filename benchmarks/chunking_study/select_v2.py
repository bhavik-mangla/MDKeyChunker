"""Apply the v2 corpus decisions and write data/v2/selection.json.

- Qasper v2: all 300 papers in data/v2/qasper_v2.json, every question.
- FreshStack v2: the 73 pilot Laravel qids (data/freshstack_laravel.json) are excluded.
  Laravel keeps its full docs corpus. Angular and YOLO keep every gold file plus a
  seeded random sample (random.Random(2027), sorted non-gold paths) of distractor
  files, round(0.5 x #gold files) per domain (DATASETS_V2.md section 3, cap B).

No LLM or network calls. Also records per-document token counts and structural
chunk counts, which orchestrate_v2.estimate() turns into cost/time estimates.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import tiktoken
from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
SELECTION = V2 / "selection.json"
PILOT_LARAVEL = ROOT / "data" / "freshstack_laravel.json"
DOMAINS = ("laravel", "angular", "yolo")
CAPPED_DOMAINS = ("angular", "yolo")
SEED = 2027
DISTRACTOR_RATIO = 0.5
ENC = tiktoken.get_encoding("cl100k_base")


def fs_doc_id(domain: str, path: str) -> str:
    return f"{domain}__{path.replace('/', '__')}"


def sample_distractors(gold_paths: list[str], all_paths: list[str], ratio: float = DISTRACTOR_RATIO,
                       seed: int = SEED) -> list[str]:
    """Seeded sample of non-gold paths, round(ratio x #gold), from the sorted pool."""
    gold = set(gold_paths)
    pool = sorted(p for p in all_paths if p not in gold)
    k = min(len(pool), round(ratio * len(gold)))
    return sorted(random.Random(seed).sample(pool, k))


def doc_stats(md: str, chunker: MarkdownChunker) -> dict:
    chunks = chunker.chunk(md)
    return {"tokens": len(ENC.encode(md)), "struct_chunks": len(chunks),
            "struct_chunk_tokens": [len(ENC.encode(c.text)) for c in chunks]}


def select(v2_dir: Path = V2, pilot_path: Path = PILOT_LARAVEL, with_stats: bool = True) -> dict:
    chunker = MarkdownChunker(Config(log_level="WARNING"))
    qasper = json.loads((v2_dir / "qasper_v2.json").read_text())
    sel: dict = {"seed": SEED, "distractor_ratio": DISTRACTOR_RATIO, "capped_domains": list(CAPPED_DOMAINS)}
    sel["qasper"] = {"paper_ids": [p["id"] for p in qasper],
                     "qids": [q["qid"] for p in qasper for q in p["questions"]]}
    doc_info: dict = {}
    if with_stats:
        for p in qasper:
            doc_info[p["id"]] = doc_stats(p["markdown"], chunker)

    pilot = {q["qid"] for q in json.loads(Path(pilot_path).read_text())["questions"]}
    docs = json.loads((v2_dir / "freshstack_v2_docs.json").read_text())
    by_id = {d["id"]: d for d in docs}
    fs: dict = {"excluded_pilot_qids": sorted(pilot)}
    for dom in DOMAINS:
        data = json.loads((v2_dir / f"freshstack_{dom}.json").read_text())
        qs = [q for q in data["questions"] if q["qid"] not in pilot]
        gold = sorted({e["path"] for q in qs for e in q["evidence"]})
        all_paths = sorted(d["path"] for d in docs if d["domain"] == dom)
        missing = [g for g in gold if g not in set(all_paths)]
        if missing:
            raise ValueError(f"{dom}: gold files missing from the docs corpus: {missing[:5]}")
        if dom in CAPPED_DOMAINS:
            distract = sample_distractors(gold, all_paths)
        else:
            distract = sorted(set(all_paths) - set(gold))
        paths = sorted(set(gold) | set(distract))
        ids = [fs_doc_id(dom, p) for p in paths]
        assert all(i in by_id for i in ids)
        fs[dom] = {"doc_ids": ids, "gold_doc_ids": [fs_doc_id(dom, p) for p in gold],
                   "distractor_doc_ids": [fs_doc_id(dom, p) for p in distract],
                   "qids": [q["qid"] for q in qs], "full_corpus": dom not in CAPPED_DOMAINS,
                   "commit": data.get("commit")}
        if with_stats:
            for i in ids:
                doc_info[i] = doc_stats(by_id[i]["markdown"], chunker)
    sel["freshstack"] = fs
    if with_stats:
        sel["doc_stats"] = {k: {"tokens": v["tokens"], "struct_chunks": v["struct_chunks"],
                                "struct_chunk_tokens": v["struct_chunk_tokens"]} for k, v in doc_info.items()}
    sel["counts"] = counts(sel)
    return sel


def counts(sel: dict) -> dict:
    ds = sel.get("doc_stats", {})

    def agg(ids: list[str], nq: int) -> dict:
        out = {"docs": len(ids), "questions": nq}
        if ds:
            out["struct_chunks"] = sum(ds[i]["struct_chunks"] for i in ids)
            out["doc_tokens"] = sum(ds[i]["tokens"] for i in ids)
            out["cr_input_tokens"] = sum(ds[i]["tokens"] * ds[i]["struct_chunks"] for i in ids)
        return out

    c = {"qasper": agg(sel["qasper"]["paper_ids"], len(sel["qasper"]["qids"]))}
    for dom in DOMAINS:
        d = sel["freshstack"][dom]
        c[f"freshstack/{dom}"] = agg(d["doc_ids"], len(d["qids"]))
    fs_ids = [i for dom in DOMAINS for i in sel["freshstack"][dom]["doc_ids"]]
    c["freshstack"] = agg(fs_ids, sum(len(sel["freshstack"][d]["qids"]) for d in DOMAINS))
    return c


def print_counts(c: dict) -> None:
    print(f"{'set':22}{'docs':>7}{'chunks':>9}{'questions':>11}{'doc tok':>11}{'CR in tok':>14}")
    for k, v in c.items():
        print(f"{k:22}{v['docs']:>7}{v.get('struct_chunks', 0):>9}{v['questions']:>11}"
              f"{v.get('doc_tokens', 0):>11,}{v.get('cr_input_tokens', 0):>14,}")


def load_selection(path: Path = SELECTION) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SELECTION))
    args = ap.parse_args()
    sel = select()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(sel))
    print_counts(sel["counts"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
