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


def _is_join(text: str, src: set) -> bool:
    """True if text is a "\n\n"-join of whole source texts (sources may contain blank lines)."""
    if not text:
        return True
    return any(text.startswith(t) and (len(text) == len(t) or text[len(t):len(t) + 2] == "\n\n")
               and _is_join(text[len(t) + 2:], src) for t in src)


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
        for e, m in (("enr_rk", "merged_rk"), ("enr_nork", "merged_nork")):
            pe, pm = CACHE / e / f"{pid}.json", CACHE / m / f"{pid}.json"
            if pe.exists() and pm.exists():
                src = {c["text"] for c in json.loads(pe.read_text())["chunks"]}
                for c in json.loads(pm.read_text())["chunks"]:
                    # merged text must be a join of source chunk texts (no LLM text injected)
                    if c["text"] not in src and not all(part in src for part in c["text"].split("\n\n") if part) \
                            and not _is_join(c["text"], src):
                        warns.append(f"{m}/{pid}: merged chunk text is not a join of {e} texts")
                        break
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
