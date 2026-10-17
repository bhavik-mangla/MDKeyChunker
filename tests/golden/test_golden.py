"""Snapshot tests: mk.chunk() on real documents (see SOURCES.md).

Any change to chunk boundaries, offsets, paths, ids or embed_text shows up as a diff in
expected/. Regenerate with ``MDK_UPDATE_GOLDEN=1`` and review the diff.
"""
import json
import os
import warnings
from pathlib import Path

import pytest

import mdkeychunker as mk

HERE = Path(__file__).parent
DOCS = sorted((HERE / "docs").iterdir())
UPDATE = os.environ.get("MDK_UPDATE_GOLDEN") == "1"


def _snapshot(doc: Path) -> str:
    text = doc.read_bytes().decode("utf-8")  # keep CRLF: offsets index the raw characters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the headerless document warns by design
        chunks = mk.chunk(text, doc_id=doc.name)
    return json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1,
                      sort_keys=True) + "\n"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_chunks_match_snapshot(doc):
    expected = HERE / "expected" / f"{doc.name}.json"
    got = _snapshot(doc)
    if UPDATE:
        expected.write_text(got, encoding="utf-8")
        return
    assert expected.exists(), f"no snapshot for {doc.name}: run with MDK_UPDATE_GOLDEN=1"
    assert got == expected.read_text(encoding="utf-8"), \
        f"{doc.name}: chunks changed; review, then MDK_UPDATE_GOLDEN=1"


def test_every_snapshot_has_a_document():
    assert {p.name for p in (HERE / "expected").iterdir()} == {f"{d.name}.json" for d in DOCS}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_golden_invariants(doc):
    """Cheap checks that hold for every snapshot, so a bad update is caught too."""
    text = doc.read_bytes().decode("utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = mk.chunk(text, doc_id=doc.name)
    assert chunks and all(c.token_count <= 512 for c in chunks)
    for c in chunks:
        lines = [ln.strip() for ln in c.text.split("\n") if ln.strip()]
        src = text[c.start_char:c.end_char]
        if not c.oversized:  # split pieces repeat fence/header rows
            assert all(ln in src for ln in lines)
