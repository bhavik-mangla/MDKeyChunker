"""Erratum (v3 PILOT, FreshStack-Laravel): zero-LLM chunk sets at the corrected commit.

For fixed512, fixedtok and struct, files whose text is unchanged between 1e8496c2 and
724c31cc get their pilot cache copied from data/cache/; changed files are re-chunked with
the same functions as build.py (no LLM). A reproducibility check first re-chunks every
pilot file at the OLD text and requires byte-identical output to the pilot cache.

For the LLM chunk sets (enr_rk, merged_rk, cr) it compares the struct chunk texts of each
changed file and writes data/cache_fixed/REBUILD_LIST.json; nothing LLM-based is run here.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build import chunk_dict, fixed_chars, fixed_tokens  # noqa: E402
from evaluate_corpus import doc_id  # noqa: E402
from mdkeychunker.chunker import MarkdownChunker  # noqa: E402
from mdkeychunker.config import Config  # noqa: E402

OLD_CACHE = ROOT / "data" / "cache"
NEW_CACHE = ROOT / "data" / "cache_fixed"
ZERO_LLM = ("fixed512", "fixedtok", "struct")
LLM_SETS = ("enr_rk", "merged_rk", "cr")


def zero_llm_sets(md: str, cfg: Config) -> dict:
    return {"fixed512": fixed_chars(md), "fixedtok": fixed_tokens(md),
            "struct": [chunk_dict(c) for c in MarkdownChunker(cfg).chunk(md)]}


def main() -> None:
    old = {d["path"]: d["markdown"] for d in json.loads((ROOT / "data" / "freshstack_laravel.json").read_text())["docs"]}
    fixed = json.loads((ROOT / "data" / "freshstack_laravel_fixed.json").read_text())
    new = {d["path"]: d["markdown"] for d in fixed["docs"]}
    # build.py's Config, minus the LLM endpoint (Stage 1 never calls it)
    cfg = Config(llm_provider="openai_compatible", llm_base_url="http://unused/v1",
                 llm_model="qwen2.5:7b", log_level="WARNING")

    # 1. reproducibility: re-chunk old text, must equal pilot cache exactly
    for path, md in old.items():
        built = zero_llm_sets(md, cfg)
        for cs in ZERO_LLM:
            cached = json.loads((OLD_CACHE / cs / f"{doc_id(path)}.json").read_text())["chunks"]
            assert built[cs] == cached, f"cannot reproduce pilot {cs} for {path}"
    print(f"reproducibility: {len(old)} files x {len(ZERO_LLM)} chunk sets identical to pilot cache")

    # 2. corrected cache: copy unchanged, rebuild changed
    changed = sorted(p for p in old if old[p] != new[p])
    report = {"changed_files": changed, "zero_llm": {}, "llm_rebuild": []}
    for cs in ZERO_LLM:
        (NEW_CACHE / cs).mkdir(parents=True, exist_ok=True)
        for path in old:
            src, dst = OLD_CACHE / cs / f"{doc_id(path)}.json", NEW_CACHE / cs / f"{doc_id(path)}.json"
            if path not in changed:
                shutil.copyfile(src, dst)
                continue
            before = json.loads(src.read_text())["chunks"]
            after = zero_llm_sets(new[path], cfg)[cs]
            dst.write_text(json.dumps({"chunks": after, "stats": {"erratum": "rebuilt at 724c31cc"}}))
            n_diff = sum(a != b for a, b in zip(before, after)) + abs(len(before) - len(after))
            report["zero_llm"][f"{cs}/{path}"] = {"chunks_before": len(before), "chunks_after": len(after),
                                                  "chunks_text_changed": n_diff}
            print(f"  {cs:9} {path}: {len(before)} -> {len(after)} chunks, {n_diff} changed")

    # 3. LLM sets: which struct chunks changed (enr/cr take struct chunks as input;
    #    rolling keys and CR's whole-document prompt make every chunk of the file depend on it)
    for path in changed:
        b = json.loads((OLD_CACHE / "struct" / f"{doc_id(path)}.json").read_text())["chunks"]
        a = json.loads((NEW_CACHE / "struct" / f"{doc_id(path)}.json").read_text())["chunks"]
        idx = [i for i, (x, y) in enumerate(zip(b, a)) if x != y]
        entry = {"doc_id": doc_id(path), "path": path, "struct_chunks_before": len(b),
                 "struct_chunks_after": len(a), "struct_chunk_indices_changed": idx,
                 "struct_chunks_changed_text": [{"index": i, "section": a[i]["section"],
                                                 "old_excerpt": None, "new_excerpt": None} for i in idx]}
        for i, e in zip(idx, entry["struct_chunks_changed_text"]):
            j = next(k for k in range(len(b[i]["text"])) if k >= len(a[i]["text"]) or b[i]["text"][k] != a[i]["text"][k])
            e["old_excerpt"], e["new_excerpt"] = b[i]["text"][max(0, j - 60):j + 40], a[i]["text"][max(0, j - 60):j + 40]
        for cs in LLM_SETS:
            n_old = len(json.loads((OLD_CACHE / cs / f"{doc_id(path)}.json").read_text())["chunks"])
            entry[f"{cs}_chunks_pilot"] = n_old
        report["llm_rebuild"].append(entry)
    (NEW_CACHE / "_erratum_chunks.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
