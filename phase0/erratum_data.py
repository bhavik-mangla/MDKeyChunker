"""Erratum (v3 PILOT, FreshStack-Laravel): rebuild the pilot dataset at the correct commit.

The pilot (prepare_freshstack.py) used laravel/docs 1e8496c2; FreshStack's corpus
matches 724c31cc (data/v2/DATASETS_V2.md). The pilot also sliced spans by BYTE offset,
but FreshStack ids carry CHARACTER offsets (they differ after non-ASCII text: releases.md).
This keeps the pilot's 24 files and 73
questions, takes file text from the verified 724c31cc copies in data/v2/raw/laravel/files
(checked here against the git blob SHA-1 in the commit tree), re-slices every gold
span from its FreshStack corpus id (character offsets) and checks it against the corpus
chunk text in data/v2/raw/corpus_laravel.parquet.

Writes data/freshstack_laravel_fixed.json and results/phase0/erratum_data.json.
"""
import difflib
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "724c31ccd3edce6b6dfe5e0dd2a594a47217f078"
RAW = ROOT / "data" / "v2" / "raw" / "laravel"
OLD = ROOT / "data" / "freshstack_laravel.json"
NEW = ROOT / "data" / "freshstack_laravel_fixed.json"
REPORT = ROOT / "results" / "phase0" / "erratum_data.json"


def blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> None:
    old = json.loads(OLD.read_text())
    tree = json.loads((RAW / f"tree_{COMMIT}_rec.json").read_text())["tree"]
    shas = {t["path"]: t["sha"] for t in tree if t["type"] == "blob"}
    corpus = {r["_id"]: r["text"] for r in pq.read_table(
        ROOT / "data" / "v2" / "raw" / "corpus_laravel.parquet", columns=["_id", "text"]).to_pylist()}

    docs, raw, file_diff = [], {}, {}
    for d in old["docs"]:
        b = (RAW / "files" / d["path"]).read_bytes()
        assert blob_sha(b) == shas[d["path"]], f"blob SHA mismatch {d['path']}"
        raw[d["path"]] = b
        md = b.decode("utf-8")
        docs.append({"path": d["path"], "markdown": md})
        if md != d["markdown"]:
            a, c = d["markdown"].splitlines(), md.splitlines()
            sm = difflib.SequenceMatcher(None, a, c, autojunk=False)
            ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
            file_diff[d["path"]] = {
                "chars_old": len(d["markdown"]), "chars_new": len(md),
                "bytes_old": len(d["markdown"].encode()), "bytes_new": len(b),
                "hunks": len(ops),
                "lines_removed": sum(i2 - i1 for _, i1, i2, _, _ in ops),
                "lines_added": sum(j2 - j1 for _, _, _, j1, j2 in ops),
                "diff": list(difflib.unified_diff(a, c, "1e8496c2", "724c31cc", n=0, lineterm="")),
            }

    questions, span_status, old_exact = [], {}, {}
    for q in old["questions"]:
        ev = []
        for e in q["evidence"]:
            # FreshStack id offsets are CHARACTER offsets into the decoded file (the pilot
            # treated them as bytes; identical unless non-ASCII text precedes the span).
            cid, md = e["corpus_id"], docs_md(docs, e["path"])
            _, cs, ce = cid.rsplit("_", 2)
            cs, ce = int(cs), int(ce)
            assert (cs, ce) == (e["start_byte"], e["end_byte"])  # pilot stored the id offsets
            text = md[cs:ce]
            s, t = len(md[:cs].encode()), len(md[:ce].encode())
            span_status[cid] = text == corpus[cid]
            old_exact[cid] = e["text"] == corpus[cid]
            ev.append({**e, "start": cs, "end": ce, "start_byte": s, "end_byte": t, "text": text,
                       "changed": text != e["text"]})
        questions.append({**q, "evidence": ev})

    affected = [q["qid"] for q in questions if any(e["changed"] for e in q["evidence"])]
    changed_spans = sorted({(e["path"], e["corpus_id"]) for q in questions for e in q["evidence"]
                            if e["changed"]})
    NEW.write_text(json.dumps({
        "source": old["source"], "commit": COMMIT,
        "erratum": "pilot freshstack_laravel.json used 1e8496c2; this copy uses 724c31cc "
                   "(FreshStack corpus commit). Same files and questions; evidence[].changed "
                   "marks spans whose text differs from the pilot.",
        "license": old["license"], "distractors": old["distractors"],
        "docs": docs, "questions": questions}, ensure_ascii=False, indent=1))

    rep = {
        "files": len(docs), "questions": len(questions),
        "unique_spans": len(span_status),
        "spans_exact_vs_corpus_fixed": sum(span_status.values()),
        "spans_exact_vs_corpus_pilot": sum(old_exact.values()),
        "changed_spans": [{"path": p, "corpus_id": c} for p, c in changed_spans],
        "affected_qids": affected,
        "files_changed": {p: {k: v for k, v in x.items() if k != "diff"} for p, x in file_diff.items()},
        "diffs": {p: x["diff"] for p, x in file_diff.items()},
    }
    REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"wrote {NEW}")
    print(f"unique gold spans: {rep['unique_spans']}; byte-exact vs FreshStack corpus: "
          f"fixed {rep['spans_exact_vs_corpus_fixed']}, pilot {rep['spans_exact_vs_corpus_pilot']}")
    print(f"changed spans: {len(changed_spans)}; affected questions: {len(affected)}")
    for p, x in file_diff.items():
        print(f"  {p}: {x['chars_old']} -> {x['chars_new']} chars ({x['chars_new'] - x['chars_old']:+d}), "
              f"{x['hunks']} hunks, -{x['lines_removed']}/+{x['lines_added']} lines")


def docs_md(docs, path):
    return next(d["markdown"] for d in docs if d["path"] == path)


if __name__ == "__main__":
    main()
