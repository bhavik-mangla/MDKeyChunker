"""The 0.4 public data contract shared by chunking, enrichment, retrieval and diagnose."""
from mdkeychunker.models import Chunk, GoldSpan, Hit, Retriever


def test_stable_id_depends_only_on_doc_offset_and_text():
    a = Chunk(text="body", doc_id="d", start_char=10, key="k1", position_index=3)
    b = Chunk(text="body", doc_id="d", start_char=10, key="k2", position_index=7, title="t")
    assert a.stable_id() == b.stable_id()
    assert Chunk(text="body", doc_id="d", start_char=11).stable_id() != a.chunk_id
    assert Chunk(text="body", doc_id="e", start_char=10).stable_id() != a.chunk_id


def test_new_fields_default_and_round_trip():
    c = Chunk(text="x")
    assert (c.doc_id, c.start_char, c.end_char, c.section_path, c.embed_text, c.oversized, c.metadata) == (
        "", 0, 0, [], "", False, {})
    c.section_path.append("Guide")
    assert Chunk(text="y").section_path == []  # no shared mutable default
    d = Chunk(text="x", section_path=["A", "B"], metadata={"src": "u"}).to_dict()
    assert d["section_path"] == ["A", "B"] and d["metadata"] == {"src": "u"}


def test_hit_goldspan_and_retriever_protocol():
    h = Hit(chunk=Chunk(text="x"), score=0.5, ranks={"bm25": 1})
    assert h.ranks["bm25"] == 1
    assert GoldSpan(doc_id="d", start=0, end=4).text == ""

    class Mine:
        def search(self, query, k):
            return ["id1"][:k]

    assert isinstance(Mine(), Retriever)
