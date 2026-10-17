"""mk.chunk(max_tokens=None) reproduces the v4 study's header chunks exactly (opt-in).

Set MDK_STUDY_DATA to the study's data/v2 directory (qasper_v2.json,
freshstack_v2_docs.json and cache/base/struct/). Every document with a cached chunk set
is re-chunked and compared text by text and section title by section title.
"""
import json
import os
import warnings
from pathlib import Path

import pytest

import mdkeychunker as mk

DATA = os.environ.get("MDK_STUDY_DATA", "")
pytestmark = pytest.mark.skipif(not DATA, reason="set MDK_STUDY_DATA to run study parity tests")


def _docs(root: Path):
    for name in ("qasper_v2.json", "freshstack_v2_docs.json"):
        for d in json.loads((root / name).read_text(encoding="utf-8")):
            cached = root / "cache" / "base" / "struct" / f"{d['id'].replace('/', '__')}.json"
            if cached.exists():
                yield d["id"], d["markdown"], json.loads(cached.read_text())["chunks"]


def test_uncapped_chunks_equal_the_study_struct_chunks():
    root = Path(DATA)
    docs = list(_docs(root))
    assert len(docs) > 600  # 300 Qasper papers + FreshStack docs
    mismatched, prefix_diff, total = [], 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for doc_id, md, ref in docs:
            got = mk.chunk(md, doc_id=doc_id, max_tokens=None, headerless="structure")
            if ([c["text"] for c in ref], [c["section"] for c in ref]) != \
                    ([c.text for c in got], [c.section_title for c in got]):
                mismatched.append(doc_id)
            prefix_diff += sum(" > ".join(c.section_path) != c.section_title for c in got)
            total += len(got)
    assert not mismatched, mismatched[:10]
    # Documented in mk.chunk: embed_text's prefix differs from the study's for 2.3% of chunks
    assert prefix_diff / total < 0.03
