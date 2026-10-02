"""Quality gates on built chunk sets. Prints one WARN line per problem; exit 1 if any.

Existence of a cache file is not enough: enrichment can silently return empty
fields, and the contextual-retrieval model can ramble instead of answering.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "cache"
RAMBLE = ("we are given", "the chunk is", "let's", "i will", "here is the", "<document>")


def check(subset: str) -> list[str]:
    items = json.loads((ROOT / subset).read_text())
    warns = []
    for it in items:
        pid = it["id"]
        struct = CACHE / "struct" / f"{pid}.json"
        n_struct = len(json.loads(struct.read_text())["chunks"]) if struct.exists() else None
        for v in ("enr_rk", "enr_nork"):
            p = CACHE / v / f"{pid}.json"
            if not p.exists():
                continue
            ch = json.loads(p.read_text())["chunks"]
            if n_struct is not None and len(ch) != n_struct:
                warns.append(f"{v}/{pid}: {len(ch)} chunks vs {n_struct} structural")
            empty = sum(1 for c in ch if not (c["key"] and c["summary"] and c["title"]))
            if empty:
                warns.append(f"{v}/{pid}: {empty}/{len(ch)} chunks missing key/title/summary")
            noq = sum(1 for c in ch if not c["questions"])
            if noq > 0.2 * len(ch):
                warns.append(f"{v}/{pid}: {noq}/{len(ch)} chunks without questions")
        p = CACHE / "cr" / f"{pid}.json"
        if p.exists():
            ch = json.loads(p.read_text())["chunks"]
            bad = [c["context"][:60] for c in ch
                   if not c.get("context") or len(c["context"]) > 700
                   or any(r in c["context"].lower()[:80] for r in RAMBLE)]
            if bad:
                warns.append(f"cr/{pid}: {len(bad)}/{len(ch)} suspicious contexts, e.g. {bad[0]!r}")
    return warns


if __name__ == "__main__":
    subsets = sys.argv[1:] or ["data/subset.json"]
    warns = [w for s in subsets for w in check(s)]
    for w in warns:
        print("WARN", w)
    print(f"{len(warns)} warnings")
    sys.exit(1 if warns else 0)
