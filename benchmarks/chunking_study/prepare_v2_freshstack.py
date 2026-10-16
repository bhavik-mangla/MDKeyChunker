"""FreshStack v2: full Markdown docs corpora for Laravel, Angular and YOLO.

Queries, nuggets and nugget-level gold corpus IDs come from the HF dataset
freshstack/queries-oct-2024 (test split per topic). FreshStack corpus IDs are
"<repo>/<path>_<start>_<end>"; the corpus (freshstack/corpus-oct-2024) stores
the chunk text and only a branch URL (".../blob/main/..."), not a commit.

Commits were identified by scoring every commit in Oct-Nov 2024 that touched
the docs root: a commit's score is the number of corpus chunks (all .md chunks
under the root, not only gold) whose text equals the file slice at that commit.
The commits below are the unique maxima (Angular: tied with its parent
63f3d0c8, whose Markdown is identical); see data/v2/DATASETS_V2.md. This
script re-checks every corpus chunk against the pinned files on each run.

Offsets: FreshStack offsets are character offsets into the decoded file; for
ASCII-only prefixes they coincide with byte offsets. A gold span is "located"
when markdown[start:end] (char offsets) equals the corpus chunk text exactly
(falling back to a byte slice, then to a unique exact occurrence elsewhere in
the file, recorded as "relocated"). Spans that cannot be located exactly are
dropped from the gold set and counted.

Kept questions: at least one located gold span in the docs-root Markdown.
Non-Markdown gold spans, and Markdown spans outside the docs root (e.g.
angular/CHANGELOG.md), are dropped from the gold set and counted per question.
Laravel questions used in the v3 pilot (data/freshstack_laravel.json) are
excluded per ANALYSIS_PLAN_v2.md section 1 and listed in "excluded_pilot_qids".

Outputs: data/v2/freshstack_<domain>.json (v3 schema + extras) and
data/v2/freshstack_v2_docs.json (all docs, id "<domain>__<path with / -> __>").
Raw downloads are cached under data/v2/raw/.
"""
import argparse
import collections
import hashlib
import html
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
import tiktoken
from mdkeychunker.chunker import MarkdownChunker
from mdkeychunker.config import Config

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2"
RAW = OUT_DIR / "raw"
HF = "https://huggingface.co/datasets/freshstack"
QUERIES_URL = HF + "/queries-oct-2024/resolve/main/{d}/test-00000-of-00001.parquet"
CORPUS_URL = HF + "/corpus-oct-2024/resolve/main/{d}/train-00000-of-00001.parquet"
SEED = 2027
N_VERIFY = 12  # random gold spans re-verified and printed per domain

DOMAINS = {
    "laravel": {"repo": "laravel/docs", "branch": "11.x",
                "commit": "724c31ccd3edce6b6dfe5e0dd2a594a47217f078",
                "id_prefix": "docs/", "root": ""},
    "angular": {"repo": "angular/angular", "branch": "main",
                "commit": "7d9b38e97bbef0c6f6e3761e7cd86457c1e0f2e1",
                "id_prefix": "angular/", "root": "adev/src/content/"},
    "yolo": {"repo": "ultralytics/ultralytics", "branch": "main",
             "commit": "8d203cf40dadb9343241d2f65bd4c84820f27430",
             "id_prefix": "ultralytics/", "root": "docs/en/"},
}

FENCE = re.compile(r"^\s*(```|~~~)", re.M)
TABLE = re.compile(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", re.M)
LIST = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+\S", re.M)


def fetch(url: str, dest: Path, headers: dict | None = None) -> bytes:
    """Download url to dest once; return bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=600, headers=headers or {})
            r.raise_for_status()
            dest.write_bytes(r.content)
            return r.content
        except requests.RequestException as e:
            print(f"  retry {attempt + 1} {url}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def gh_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def md_tree(domain: str, cfg: dict) -> dict[str, str]:
    """{path: blob sha} of all .md files under the docs root at the commit.
    Walks down to the root sub-tree so the recursive listing is not truncated."""
    api = f"https://api.github.com/repos/{cfg['repo']}/git/trees/"
    sha, prefix = cfg["commit"], ""
    for part in [p for p in cfg["root"].split("/") if p]:
        t = json.loads(fetch(api + sha, RAW / domain / f"tree_{sha}.json", gh_headers()))
        sha = next(e["sha"] for e in t["tree"] if e["path"] == part and e["type"] == "tree")
        prefix += part + "/"
    t = json.loads(fetch(api + sha + "?recursive=1", RAW / domain / f"tree_{sha}_rec.json", gh_headers()))
    assert not t.get("truncated"), f"{domain}: tree listing truncated"
    return {prefix + e["path"]: e["sha"] for e in t["tree"]
            if e["type"] == "blob" and e["path"].endswith(".md")}


def git_blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def parse_id(cid: str) -> tuple[str, int, int]:
    path, s, e = cid.rsplit("_", 2)
    return path, int(s), int(e)


def to_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"[ \t]+", " ", html.unescape(s)).strip()


def locate(md: str, raw: bytes, s: int, e: int, text: str):
    """Return (char_start, char_end, how) with md[start:end] == text, or None."""
    if md[s:e] == text:
        return s, e, "offset_char"
    if raw[s:e].decode("utf-8", "replace") == text:
        cs = len(raw[:s].decode("utf-8")); return cs, cs + len(text), "offset_byte"
    i = md.find(text)
    if text and i >= 0 and md.find(text, i + 1) < 0:
        return i, i + len(text), "relocated"
    return None


def build_domain(domain: str, cfg: dict, pilot_qids: set[str], chunker, enc) -> tuple[dict, dict]:
    print(f"\n=== {domain}: {cfg['repo']}@{cfg['commit'][:12]} root={cfg['root'] or '/'}")
    tree = md_tree(domain, cfg)
    raw_url = f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['commit']}/{{path}}"
    raw, md = {}, {}
    for path in sorted(tree):
        b = fetch(raw_url.format(path=path), RAW / domain / "files" / path)
        if git_blob_sha(b) != tree[path]:
            raise RuntimeError(f"{domain}/{path}: blob sha mismatch")
        raw[path], md[path] = b, b.decode("utf-8")

    # corpus chunks of docs-root Markdown: verify every one against pinned files
    fetch(CORPUS_URL.format(d=domain), RAW / f"corpus_{domain}.parquet")
    corpus = {}
    for r in pq.read_table(RAW / f"corpus_{domain}.parquet", columns=["_id", "text"]).to_pylist():
        corpus[r["_id"]] = r["text"]
    pre = cfg["id_prefix"] + cfg["root"]
    how = collections.Counter()
    corpus_md_files = set()
    unlocatable_chunks = []
    for cid, text in corpus.items():
        p, s, e = parse_id(cid)
        if not (p.startswith(pre) and p.endswith(".md")):
            continue
        path = p[len(cfg["id_prefix"]):]
        corpus_md_files.add(path)
        if path not in md:
            how["file_missing"] += 1
            continue
        loc = locate(md[path], raw[path], s, e, text)
        how[loc[2] if loc else "unlocatable"] += 1
        if not loc:
            unlocatable_chunks.append(cid)
    verify = {"corpus_md_chunks": sum(how.values()), "by_method": dict(how),
              "unlocatable_chunks": unlocatable_chunks,
              "md_files_at_commit": len(md), "md_files_in_corpus": len(corpus_md_files),
              "files_only_at_commit": sorted(set(md) - corpus_md_files),
              "files_only_in_corpus": sorted(corpus_md_files - set(md))}
    print(f"corpus check: {verify['corpus_md_chunks']} chunks {dict(how)}; files "
          f"commit={len(md)} corpus={len(corpus_md_files)}")

    # questions
    fetch(QUERIES_URL.format(d=domain), RAW / f"queries_{domain}.parquet")
    rows = pq.read_table(RAW / f"queries_{domain}.parquet").to_pylist()
    out_q, dropped_q, excluded_pilot = [], [], []
    for r in rows:
        ev, ev_index, nug_ev = [], {}, []
        drop = collections.Counter()
        for nug in r["nuggets"]:
            idx = []
            for cid in nug["relevant_corpus_ids"]:
                p, s, e = parse_id(cid)
                if not p.endswith(".md"):
                    drop["non_markdown"] += 1
                    continue
                if not p.startswith(pre):
                    drop["markdown_outside_root"] += 1
                    continue
                path = p[len(cfg["id_prefix"]):]
                if cid not in ev_index:
                    text = corpus.get(cid)
                    loc = locate(md[path], raw[path], s, e, text) if text is not None and path in md else None
                    if loc is None:
                        drop["markdown_unlocatable"] += 1
                        ev_index[cid] = None
                    else:
                        cs, ce, h = loc
                        ev_index[cid] = len(ev)
                        ev.append({"path": path, "start": cs, "end": ce,
                                   "start_byte": len(md[path][:cs].encode()),
                                   "end_byte": len(md[path][:ce].encode()),
                                   "corpus_id": cid, "located": h, "text": text})
                if ev_index[cid] is not None:
                    idx.append(ev_index[cid])
            nug_ev.append(sorted(set(idx)))
        title = to_text(r["query_title"])
        rec = {"qid": r["query_id"], "title": title,
               "question": f"{title}\n\n{to_text(r['query_text'])}",
               "nuggets": [n["text"] for n in r["nuggets"]],
               "nugget_evidence": nug_ev,
               "evidence": ev,
               "gold_dropped": dict(drop),
               "all_gold_markdown_in_root": not drop,
               "answer_id": r.get("answer_id"),
               "answer": r.get("answer_text"),
               "tags": (r.get("metadata") or {}).get("tags")}
        if not ev:
            dropped_q.append(r["query_id"])
        elif r["query_id"] in pilot_qids:
            excluded_pilot.append(r["query_id"])
        else:
            out_q.append(rec)

    docs = [{"path": p, "markdown": md[p]} for p in sorted(md)]
    gold_files = sorted({e["path"] for q in out_q for e in q["evidence"]})
    out = {"source": f"freshstack/queries-oct-2024 ({domain}) + {cfg['repo']}",
           "repo": cfg["repo"], "branch": cfg["branch"], "commit": cfg["commit"],
           "docs_root": cfg["root"] or "/",
           "license": "FreshStack CC-BY-SA-4.0; docs under the repository's license",
           "corpus_verification": verify,
           "questions_total": len(rows),
           "dropped_no_markdown_gold": dropped_q,
           "excluded_pilot_qids": excluded_pilot,
           "gold_files": gold_files,
           "docs": docs, "questions": out_q}

    # random byte-exact re-verification of gold spans
    spans = [e for q in out_q for e in q["evidence"]]
    rng = random.Random(SEED)
    ok = 0
    sample = rng.sample(spans, min(N_VERIFY, len(spans)))
    for e in sample:
        b = raw[e["path"]]
        same = (b[e["start_byte"]:e["end_byte"]] == e["text"].encode()
                and md[e["path"]][e["start"]:e["end"]] == corpus[e["corpus_id"]])
        ok += same
    print(f"random gold-span verification: {ok}/{len(sample)} byte-exact")
    all_ok = all(raw[e["path"]][e["start_byte"]:e["end_byte"]] == corpus[e["corpus_id"]].encode()
                 for e in spans)

    # stats
    n_chunks = cr_tok = 0
    big16 = big32 = 0
    for d in docs:
        k = len(chunker.chunk(d["markdown"]))
        t = len(enc.encode(d["markdown"]))
        n_chunks += k
        cr_tok += k * t
        big16 += t > 16384
        big32 += t > 32768
    uniq = {e["corpus_id"]: e["text"] for e in spans}
    drops = collections.Counter()
    for q in out_q:
        drops.update(q["gold_dropped"])
    st = {"files": len(docs),
          "kb": round(sum(len(d["markdown"].encode()) for d in docs) / 1024, 1),
          "structural_chunks": n_chunks,
          "doc_tokens": sum(len(enc.encode(d["markdown"])) for d in docs),
          "cr_input_tokens": cr_tok,
          "docs_over_16k_tokens": big16, "docs_over_32k_tokens": big32,
          "questions_total": len(rows), "questions_kept": len(out_q),
          "dropped_no_markdown_gold": len(dropped_q),
          "excluded_pilot": len(excluded_pilot),
          "kept_all_gold_markdown_in_root": sum(q["all_gold_markdown_in_root"] for q in out_q),
          "gold_spans_kept": len(spans), "unique_gold_spans": len(uniq),
          "gold_files": len(gold_files),
          "spans_per_q_mean": round(statistics.mean(len(q["evidence"]) for q in out_q), 2),
          "gold_dropped_in_kept_questions": dict(drops),
          "located": dict(collections.Counter(e["located"] for e in spans)),
          "unique_spans_with_code_fence": sum(bool(FENCE.search(t)) for t in uniq.values()),
          "unique_spans_with_table": sum(bool(TABLE.search(t)) for t in uniq.values()),
          "unique_spans_with_list": sum(bool(LIST.search(t)) for t in uniq.values()),
          "random_verify": f"{ok}/{len(sample)}", "all_gold_byte_exact": all_ok,
          "corpus_chunks_located": f"{verify['corpus_md_chunks'] - how['unlocatable'] - how['file_missing']}"
                                   f"/{verify['corpus_md_chunks']}"}
    out["stats"] = st
    for k, v in st.items():
        print(f"  {k}: {v}")
    return out, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="laravel,angular,yolo")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pilot = {q["qid"] for q in json.loads((ROOT / "data" / "freshstack_laravel.json").read_text())["questions"]}
    chunker = MarkdownChunker(Config(log_level="WARNING"))
    enc = tiktoken.get_encoding("cl100k_base")
    all_docs, stats = [], {}
    for domain in args.domains.split(","):
        out, st = build_domain(domain, DOMAINS[domain], pilot if domain == "laravel" else set(),
                               chunker, enc)
        (OUT_DIR / f"freshstack_{domain}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        stats[domain] = st
        for d in out["docs"]:
            all_docs.append({"id": f"{domain}__{d['path'].replace('/', '__')}",
                             "domain": domain, "path": d["path"], "markdown": d["markdown"]})
    ids = [d["id"] for d in all_docs]
    assert len(ids) == len(set(ids))
    (OUT_DIR / "freshstack_v2_docs.json").write_text(json.dumps(all_docs, ensure_ascii=False, indent=1))
    (OUT_DIR / "freshstack_v2_stats.json").write_text(json.dumps(stats, indent=1))
    tot = {k: sum(s[k] for s in stats.values())
           for k in ("files", "structural_chunks", "cr_input_tokens", "questions_kept")}
    print(f"\nTOTAL {tot}; docs -> {OUT_DIR / 'freshstack_v2_docs.json'}")


if __name__ == "__main__":
    main()
