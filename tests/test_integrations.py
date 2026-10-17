"""LangChain / LlamaIndex adapters and to_chunks coercion."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import mdkeychunker as mk
from mdkeychunker.integrations import to_chunks
from mdkeychunker.models import Chunk

DOC = """# Guide

Intro paragraph that explains what the guide covers and why it exists here.

## Install

Run the installer and follow the prompts. It takes a minute on most machines.

```bash
pip install example
```

## Usage

- first item in a list
- second item in a list
"""
ROOT = Path(__file__).resolve().parents[1]


def lines_within(text, src):
    return all(ln.strip() in src for ln in text.split("\n") if ln.strip())


# --------------------------------------------------------------------------- #
# to_chunks (no dependency)                                                   #
# --------------------------------------------------------------------------- #

def test_to_chunks_strings_dicts_and_chunks():
    own = mk.chunk(DOC, doc_id="g")[0]
    cs = to_chunks(["plain", {"page_content": "lc-like", "metadata": {"source": "s.md",
                                                                      "start_index": 7}},
                    {"text": "t", "id": "my-id", "doc_id": "d", "start_char": 3},
                    {"content": "c", "section_path": ["A", "B"]}, own])
    assert [c.text for c in cs] == ["plain", "lc-like", "t", "c", own.text]
    assert cs[1].doc_id == "s.md" and cs[1].start_char == 7 and cs[1].end_char == 14
    assert cs[2].chunk_id == "my-id" and cs[2].doc_id == "d" and cs[2].start_char == 3
    assert cs[3].section_path == ["A", "B"] and cs[3].embed_text == "c"
    assert cs[4] is own
    assert cs[0].chunk_id == to_chunks(["plain"])[0].chunk_id  # stable without an id


def test_to_chunks_chonkie_style_and_errors():
    chonkie = types.SimpleNamespace(text="hello", start_index=5, end_index=10, token_count=2)
    c = to_chunks([chonkie])[0]
    assert (c.text, c.start_char, c.end_char) == ("hello", 5, 10)
    with pytest.raises(TypeError):
        to_chunks([42])
    with pytest.raises(TypeError):
        to_chunks([{"title": "no text"}])


def test_to_chunks_feeds_evidence_recall():
    cs = to_chunks([{"page_content": "the answer is 42 according to the docs",
                     "metadata": {"source": "d"}}])
    assert mk.evidence_recall(cs, ["the answer is 42 according to the docs"]) == 1.0


def test_integrations_package_imports_no_framework():
    code = ("import sys, json, mdkeychunker.integrations as i; i.to_chunks(['x']); "
            "print(json.dumps([m for m in ('langchain_core', 'langchain_text_splitters', "
            "'llama_index') if m in sys.modules]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", check=True)
    assert json.loads(out.stdout) == []


# --------------------------------------------------------------------------- #
# LangChain                                                                   #
# --------------------------------------------------------------------------- #

def _lc():
    pytest.importorskip("langchain_text_splitters")
    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter

    from mdkeychunker.integrations.langchain import MDKeyChunkerSplitter
    return types.SimpleNamespace(Document=Document, TextSplitter=TextSplitter,
                                 Splitter=MDKeyChunkerSplitter)


def test_langchain_split_documents_matches_mk_chunk():
    lc = _lc()
    splitter = lc.Splitter(max_tokens=64, headerless="structure")
    assert isinstance(splitter, lc.TextSplitter)
    src = lc.Document(page_content=DOC, metadata={"source": "guide.md", "lang": "en"})
    docs = splitter.split_documents([src])
    expected = mk.chunk(DOC, doc_id="guide.md", max_tokens=64, headerless="structure")
    assert [d.page_content for d in docs] == [c.embed_text for c in expected]
    assert [d.id for d in docs] == [c.chunk_id for c in expected]
    for d, c in zip(docs, expected):
        assert d.metadata["source"] == "guide.md" and d.metadata["lang"] == "en"
        assert d.metadata["section"] == " > ".join(c.section_path)
        assert (d.metadata["start_index"], d.metadata["end_index"]) == (c.start_char, c.end_char)
        if not c.oversized:
            assert lines_within(c.text, DOC[d.metadata["start_index"]:d.metadata["end_index"]])
        assert all(isinstance(v, (str, int, bool)) for v in d.metadata.values())
    again = splitter.transform_documents([src])
    assert [d.id for d in again] == [d.id for d in docs]


def test_langchain_text_mode_split_text_and_create_documents():
    lc = _lc()
    raw = lc.Splitter(content="text")
    assert raw.split_text(DOC) == [c.text for c in mk.chunk(DOC)]
    docs = raw.create_documents([DOC, DOC], metadatas=[{"source": "a"}, {"source": "b"}])
    assert {d.metadata["source"] for d in docs} == {"a", "b"}
    assert len({d.id for d in docs}) == len(docs)  # doc_id is part of the id
    with pytest.raises(ValueError):
        lc.Splitter(content="html")


def test_langchain_documents_coerce_back():
    lc = _lc()
    docs = lc.Splitter().split_documents([lc.Document(page_content=DOC, metadata={"source": "g"})])
    back = to_chunks(docs)
    assert [c.chunk_id for c in back] == [d.id for d in docs]
    assert [c.start_char for c in back] == [d.metadata["start_index"] for d in docs]


# --------------------------------------------------------------------------- #
# LlamaIndex                                                                  #
# --------------------------------------------------------------------------- #

def _li():
    pytest.importorskip("llama_index.core")
    from llama_index.core import Document
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import MetadataMode, NodeRelationship

    from mdkeychunker.integrations.llamaindex import MDKeyChunkerNodeParser
    return types.SimpleNamespace(Document=Document, NodeParser=NodeParser, Mode=MetadataMode,
                                 Rel=NodeRelationship, Parser=MDKeyChunkerNodeParser)


def test_llamaindex_nodes_embed_exactly_the_mk_embed_text():
    li = _li()
    parser = li.Parser(max_tokens=64, headerless="structure")
    assert isinstance(parser, li.NodeParser)
    doc = li.Document(text=DOC, id_="guide")
    nodes = parser.get_nodes_from_documents([doc])
    expected = mk.chunk(DOC, doc_id="guide", max_tokens=64, headerless="structure")
    assert [n.node_id for n in nodes] == [c.chunk_id for c in expected]
    assert [n.get_content(metadata_mode=li.Mode.EMBED) for n in nodes] == \
        [c.embed_text for c in expected]
    assert [n.get_content(metadata_mode=li.Mode.LLM) for n in nodes] == \
        [c.embed_text for c in expected]
    for n, c in zip(nodes, expected):
        assert (n.start_char_idx, n.end_char_idx) == (c.start_char, c.end_char)
        assert n.ref_doc_id == "guide" and n.metadata["section"] == " > ".join(c.section_path)
    assert nodes[0].relationships[li.Rel.NEXT].node_id == nodes[1].node_id
    assert nodes[1].relationships[li.Rel.PREVIOUS].node_id == nodes[0].node_id


def test_llamaindex_text_mode_and_document_metadata():
    li = _li()
    doc = li.Document(text=DOC, id_="g", metadata={"file_name": "guide.md"})
    nodes = li.Parser(content="text", max_tokens=64).get_nodes_from_documents([doc])
    first = nodes[1]
    embedded = first.get_content(metadata_mode=li.Mode.EMBED)
    assert "file_name: guide.md" in embedded and f"section: {first.metadata['section']}" in embedded
    assert first.get_content(metadata_mode=li.Mode.NONE) == \
        mk.chunk(DOC, max_tokens=64)[1].text
    assert "chunk_id" not in embedded
    back = to_chunks(nodes)
    assert [c.chunk_id for c in back] == [n.node_id for n in nodes]
    assert back[1].doc_id == "g" and back[1].embed_text == embedded
    with pytest.raises(ValueError):
        li.Parser(content="html").get_nodes_from_documents([doc])


def test_coerced_types_are_chunks():
    assert all(isinstance(c, Chunk) for c in to_chunks(["a", {"text": "b"}]))
