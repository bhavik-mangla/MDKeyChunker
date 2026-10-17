"""diagnose accepts other libraries' chunks as the corpus and as retriever results."""
import sys
import types
from enum import Enum

import pytest

from mdkeychunker.diagnose import diagnose

pytest.importorskip("numpy")  # the built-in dense half needs numpy

N_DOCS = 12


def section(d, s):
    return f"Section {s} of manual {d} explains the topic word zq{d}x{s} in plain detail."


def qa():
    return [{"question": f"what does zq{d}x{s} mean?", "gold": [section(d, s)], "doc_id": f"d{d}"}
            for d in range(N_DOCS) for s in (1, 3)]


def flat(texts):
    return [[1.0, 0.0] for _ in texts]


KW = dict(bm25="study", embedder=flat, budget_tokens=19, n_boot=300)


def topic(query):
    word = next(w for w in query.replace("?", "").split() if w.startswith("zq"))
    d, s = word[2:].split("x")
    return int(d), int(s)


# --------------------------------------------------------------------------- fakes (duck-typed)

class LCDocument:
    """Shaped like langchain_core.documents.Document."""

    def __init__(self, page_content, metadata=None, id=None):
        self.page_content, self.metadata, self.id = page_content, metadata or {}, id


class ChonkieChunk:
    """Shaped like a Chonkie chunk: text, start_index, end_index (no doc id)."""

    def __init__(self, text, start_index, end_index, id=None):
        self.text, self.start_index, self.end_index, self.id = text, start_index, end_index, id


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    NONE = "none"


class LINode:
    """Shaped like a llama_index TextNode."""

    def __init__(self, text, node_id, ref_doc_id, metadata=None):
        self.text, self.node_id, self.ref_doc_id = text, node_id, ref_doc_id
        self.metadata = metadata or {}
        self.start_char_idx = self.end_char_idx = None

    def get_content(self, metadata_mode=MetadataMode.NONE):
        if metadata_mode == MetadataMode.EMBED and self.metadata.get("section"):
            return f"{self.metadata['section']}\n\n{self.text}"
        return self.text


@pytest.fixture
def stub_llamaindex(monkeypatch):
    """to_chunks imports llama_index.core.schema.MetadataMode for nodes; stub it if absent."""
    try:
        import llama_index.core.schema  # noqa: F401
        return
    except ImportError:
        pass
    schema = types.ModuleType("llama_index.core.schema")
    schema.MetadataMode = MetadataMode
    core = types.ModuleType("llama_index.core")
    core.schema = schema
    root = types.ModuleType("llama_index")
    root.core = core
    monkeypatch.setitem(sys.modules, "llama_index", root)
    monkeypatch.setitem(sys.modules, "llama_index.core", core)
    monkeypatch.setitem(sys.modules, "llama_index.core.schema", schema)


def lc_corpus():
    return {(d, s): LCDocument(section(d, s), {"source": f"d{d}", "chunk_id": f"lc-{d}-{s}"})
            for d in range(N_DOCS) for s in range(4)}


# --------------------------------------------------------------------------- tests

def test_langchain_documents_as_corpus_and_as_retriever_results():
    docs = lc_corpus()

    def lc_search(query, k):  # returns LangChain Documents: another doc's decoy, then the gold
        d, s = topic(query)
        return [docs[((d + 1) % N_DOCS, s)], docs[(d, s)]][:k]

    r = diagnose(list(docs.values()), qa(), retrievers={"lc": lc_search}, check=("fusion",),
                 compare=[("lc", "bm25")], **KW)
    assert r.settings["scope"] == "document" and r.settings["n_clusters"] == N_DOCS
    assert r.arms["bm25"].mean == 1.0  # corpus chunks kept doc_id from metadata["source"]
    assert r.arms["lc"].mean == 1.0    # the decoy from another document was dropped by doc_id
    assert {(f.a, f.b) for f in r.findings} >= {("hybrid", "dense"), ("lc", "bm25")}


def test_retriever_results_resolve_to_the_corpus_chunk_by_id():
    docs = lc_corpus()
    seen = []

    def lc_search(query, k):  # same ids, but no source metadata: only the id links them
        d, s = topic(query)
        hit = LCDocument(section(d, s), {"chunk_id": f"lc-{d}-{s}"})
        seen.append(hit)
        return [hit]

    r = diagnose(list(docs.values()), qa(), retrievers={"lc": lc_search}, check=(), **KW)
    assert r.arms["lc"].mean == 1.0 and len(seen) == len(qa())


def test_llamaindex_nodes_use_embed_mode_text_for_indexing(stub_llamaindex):
    nodes = [LINode(section(d, s), f"n-{d}-{s}", f"d{d}", {"section": f"Topic zq{d}x{s}"})
             for d in range(N_DOCS) for s in range(4)]
    by_id = {n.node_id: n for n in nodes}

    def li_search(query, k):
        d, s = topic(query)
        return [by_id[f"n-{d}-{s}"]]

    r = diagnose(nodes, qa(), retrievers={"li": li_search}, check=("fusion",), **KW)
    assert r.arms["bm25"].mean == 1.0 and r.arms["li"].mean == 1.0


def test_chonkie_chunks_with_offsets_and_offset_gold():
    # Chonkie chunks carry offsets but no doc id: corpus scope, offset gold in one document
    chunks = [ChonkieChunk(section(0, s), 100 * s, 100 * s + len(section(0, s)), id=f"c{s}")
              for s in range(4)]
    qs = [{"question": f"what does zq0x{s} mean?", "gold": [("", 100 * s, 100 * s + 30)]}
          for s in range(4)]
    r = diagnose(chunks, qs, check=(), compare=[("bm25", "bm25")], **KW)
    assert r.settings["scope"] == "corpus"
    assert r.arms["bm25"].mean == 1.0


def test_mixed_and_unknown_items():
    docs = lc_corpus()
    extra = {"page_content": section(5, 0), "metadata": {"source": "d5"}}
    mixed = list(docs.values())[:8] + [extra]
    r = diagnose(mixed, [{"question": "what does zq0x1 mean?", "gold": [section(0, 1)],
                          "doc_id": "d0"}], check=(), compare=[("bm25", "bm25")], **KW)
    assert r.arms["bm25"].mean == 1.0
    with pytest.raises(TypeError, match=r"\{doc_id: markdown\}"):
        diagnose([object()], qa(), **KW)

    def bad(query, k):
        return [3.5]

    with pytest.raises(TypeError, match="a retriever returned float"):
        diagnose(list(docs.values()), qa(), retrievers={"bad": bad}, check=(), **KW)


def test_mdkeychunker_splitter_output_end_to_end():
    pytest.importorskip("langchain_text_splitters")
    from langchain_core.documents import Document

    from mdkeychunker.integrations.langchain import MDKeyChunkerSplitter

    md = {f"d{d}": "\n\n".join(f"# S{s}\n\n{section(d, s)}" for s in range(8))
          for d in range(N_DOCS)}
    split = MDKeyChunkerSplitter().split_documents(
        [Document(page_content=text, metadata={"source": doc_id}) for doc_id, text in md.items()])
    r = diagnose(split, qa(), check=("fusion",), **{**KW, "budget_tokens": 45})
    assert r.settings["n_clusters"] == N_DOCS and r.arms["bm25"].mean == 1.0


def test_real_llamaindex_text_nodes():
    schema = pytest.importorskip("llama_index.core.schema")
    nodes = [schema.TextNode(text=section(d, s), id_=f"n-{d}-{s}",
                             metadata={"doc_id": f"d{d}"})
             for d in range(N_DOCS) for s in range(4)]
    by_id = {n.node_id: n for n in nodes}

    def li_search(query, k):
        d, s = topic(query)
        return [by_id[f"n-{d}-{s}"]]

    r = diagnose(nodes, qa(), retrievers={"li": li_search}, check=(), **KW)
    assert r.arms["li"].mean == 1.0
