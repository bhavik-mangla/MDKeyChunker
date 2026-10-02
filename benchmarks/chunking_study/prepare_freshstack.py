"""Build a FreshStack-Laravel retrieval subset over native Markdown docs.

Queries + nugget-level relevance: HF freshstack/queries-oct-2024 (laravel/test).
Documents: laravel/docs at a pinned commit; FreshStack corpus IDs are
"<repo>/<path>_<start_byte>_<end_byte>" and the byte ranges match this commit
exactly, so gold evidence is recovered as file bytes without downloading the
39 MB corpus parquet. Raw downloads are cached under data/freshstack_raw/.
"""
import html
import json
import re
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests

COMMIT = "1e8496c232c2e3f17179c7ad8d751f9c795ce16a"
QUERIES_URL = ("https://huggingface.co/datasets/freshstack/queries-oct-2024/"
               "resolve/main/laravel/test-00000-of-00001.parquet")
TREE_URL = f"https://api.github.com/repos/laravel/docs/git/trees/{COMMIT}"
RAW_URL = f"https://raw.githubusercontent.com/laravel/docs/{COMMIT}/{{path}}"
FILTER_URL = "https://datasets-server.huggingface.co/filter"
BUDGET = 600_000  # bytes of Markdown
N_DISTRACTORS = 5
N_SPOTCHECK = 6

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "freshstack_raw"
OUT = ROOT / "data" / "freshstack_laravel.json"
DOC_PREFIX = "docs/"  # laravel/docs repo prefix inside FreshStack IDs


def fetch(url, dest, **kw):
    """Download url to dest once; return bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=600, **kw)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return r.content
        except requests.RequestException as e:
            print(f"  retry {attempt + 1} {url}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def parse_id(cid):
    path, s, e = cid.rsplit("_", 2)
    return path, int(s), int(e)


def to_text(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"[ \t]+", " ", html.unescape(s)).strip()


def select_files(questions, sizes):
    """Greedy max-coverage: repeatedly add the file set that completes the most
    questions per added byte, until the byte budget is exhausted."""
    chosen, used = set(), 0
    while True:
        best = None
        for q in questions:
            need = q["files"] - chosen
            if not need:
                continue
            cost = sum(sizes[f] for f in need)
            if used + cost > BUDGET:
                continue
            gain = sum(1 for o in questions
                       if not (o["files"] - chosen - need)
                       and (o["files"] - chosen))
            score = gain / cost
            if best is None or score > best[0]:
                best = (score, need, cost)
        if best is None:
            return chosen, used
        chosen |= best[1]
        used += best[2]


def spot_check(evidence_by_id, n):
    """Compare our byte slices with FreshStack corpus chunk text (HF filter API)."""
    ok, checked = 0, 0
    for cid, text in list(evidence_by_id.items())[:n * 3]:
        if checked >= n:
            break
        cache = RAW / f"corpus_{re.sub(r'[^A-Za-z0-9_.-]', '_', cid)}.json"
        params = {"dataset": "freshstack/corpus-oct-2024", "config": "laravel",
                  "split": "train", "where": f"\"_id\"='{cid}'", "length": 1}
        try:
            row = None
            for _ in range(4):
                data = json.loads(fetch(FILTER_URL, cache, params=params))
                if data.get("rows"):
                    row = data["rows"][0]["row"]
                    break
                cache.unlink(missing_ok=True)  # index loading / error: retry
                time.sleep(15)
            if row is None:
                continue
        except Exception as e:  # network/index hiccup: skip this one
            print(f"  spot-check skipped {cid}: {e}", file=sys.stderr)
            continue
        checked += 1
        same = row["text"] == text
        ok += same
        print(f"  spot-check {cid}: {'OK' if same else 'MISMATCH'}")
    return ok, checked


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. queries + nugget-level gold chunk IDs
    fetch(QUERIES_URL, RAW / "laravel_queries.parquet")
    rows = pq.read_table(RAW / "laravel_queries.parquet").to_pylist()
    print(f"queries: {len(rows)}")

    tree = json.loads(fetch(TREE_URL, RAW / "tree.json"))["tree"]
    sizes = {t["path"]: t["size"] for t in tree
             if t["type"] == "blob" and t["path"].endswith(".md")}

    # 2. Markdown evidence from laravel/docs only; gold = union over nuggets
    questions, no_md = [], 0
    for r in rows:
        spans = set()
        for nug in r["nuggets"]:
            for cid in nug["relevant_corpus_ids"]:
                path, s, e = parse_id(cid)
                if path.startswith(DOC_PREFIX) and path[len(DOC_PREFIX):] in sizes:
                    spans.add((path[len(DOC_PREFIX):], s, e, cid))
        if not spans:
            no_md += 1
            continue
        questions.append({"row": r, "spans": sorted(spans),
                          "files": {p for p, *_ in spans}})
    print(f"with laravel/docs Markdown evidence: {len(questions)} "
          f"(dropped {no_md} without)")

    chosen, used = select_files(questions, sizes)
    kept = [q for q in questions if q["files"] <= chosen]

    # distractors: smallest unused, non-boilerplate files that still fit
    skip = {"readme.md", "license.md", "documentation.md"}
    for path in sorted((p for p in sizes if p not in chosen and p not in skip),
                       key=lambda p: sizes[p]):
        if len(chosen - {f for q in kept for f in q["files"]}) >= N_DISTRACTORS:
            break
        if used + sizes[path] <= BUDGET:
            chosen.add(path)
            used += sizes[path]
    distractors = sorted(chosen - {f for q in kept for f in q["files"]})

    # 3. fetch docs, slice evidence (byte offsets -> char offsets)
    docs, raw_bytes = [], {}
    for path in sorted(chosen):
        b = fetch(RAW_URL.format(path=path), RAW / path)
        if len(b) != sizes[path]:
            raise RuntimeError(f"size mismatch for {path}: {len(b)} vs {sizes[path]}")
        raw_bytes[path] = b
        docs.append({"path": path, "markdown": b.decode("utf-8")})

    out_q, evidence_by_id = [], {}
    for q in kept:
        r = q["row"]
        ev = []
        for path, s, e, cid in q["spans"]:
            b = raw_bytes[path]
            cs, ce = len(b[:s].decode("utf-8")), len(b[:e].decode("utf-8"))
            text = b[s:e].decode("utf-8")
            assert b.decode("utf-8")[cs:ce] == text
            ev.append({"path": path, "start": cs, "end": ce,
                       "start_byte": s, "end_byte": e,
                       "corpus_id": cid, "text": text})
            evidence_by_id.setdefault(cid, text)
        title = to_text(r["query_title"])
        out_q.append({"qid": r["query_id"], "title": title,
                      "question": f"{title}\n\n{to_text(r['query_text'])}",
                      "nuggets": [n["text"] for n in r["nuggets"]],
                      "evidence": ev})

    ok, checked = spot_check(evidence_by_id, N_SPOTCHECK)

    OUT.write_text(json.dumps({
        "source": "freshstack/queries-oct-2024 (laravel) + laravel/docs",
        "commit": COMMIT,
        "license": "FreshStack CC-BY-SA-4.0; laravel/docs MIT",
        "distractors": distractors,
        "docs": docs,
        "questions": out_q}, ensure_ascii=False, indent=1))

    # 4. stats
    spans = [e for q in out_q for e in q["evidence"]]
    uniq = {e["corpus_id"]: e["text"] for e in spans}
    fence = sum("```" in t for t in uniq.values())
    table = sum(bool(re.search(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", t, re.M))
                for t in uniq.values())
    lists = sum(bool(re.search(r"^\s*(?:[-*+]|\d+\.)\s+\S", t, re.M))
                for t in uniq.values())
    per_q = [len(q["evidence"]) for q in out_q]
    print(f"\nwrote {OUT}")
    print(f"docs: {len(docs)} ({len(distractors)} distractors), "
          f"{sum(len(d['markdown'].encode()) for d in docs) / 1024:.1f} KB")
    print(f"questions kept: {len(out_q)} of {len(questions)} with Markdown evidence "
          f"({len(rows)} total)")
    print(f"evidence spans/question: mean {statistics.mean(per_q):.2f}, "
          f"median {statistics.median(per_q)}, max {max(per_q)}")
    print(f"unique evidence spans: {len(uniq)}; median length "
          f"{statistics.median(len(t) for t in uniq.values()):.0f} chars / "
          f"{statistics.median(len(t.split()) for t in uniq.values()):.0f} words")
    print(f"spans with code fences: {fence}, tables: {table}, lists: {lists}")
    print(f"spot-check vs corpus text: {ok}/{checked} identical")
    print("files:", ", ".join(d["path"] for d in docs))


if __name__ == "__main__":
    main()
