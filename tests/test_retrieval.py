"""Reference retrievers with fake embedders/rerankers (no model download)."""
import sys

import pytest

from mdkeychunker.models import Chunk, Hit, Retriever
from mdkeychunker.retrieval import (
    RRF,
    BM25Retriever,
    Convex,
    DenseRetriever,
    HybridRetriever,
    Reranker,
)
from mdkeychunker.retrieval.rerank import RerankCache, format_pair, rerank_order

np = pytest.importorskip("numpy")

DOCS = [
    "Install the package on Linux with pip.",
    "Pricing plans and billing for teams.",
    "Configure the Linux service and restart it.",
    "",
    "Troubleshooting install errors on Windows.",
]


def chunks():
    return [Chunk(text=t, chunk_id=f"c{i}") for i, t in enumerate(DOCS)]


VOCAB = ["linux", "install", "pricing", "windows", "service"]


def bag(texts):
    """Toy embedder: bag of a few words plus a constant so no vector is zero."""
    return [[float(w in t.lower()) for w in VOCAB] + [0.1] for t in texts]


# --------------------------------------------------------------------------- BM25

@pytest.fixture
def bm25s_installed():
    pytest.importorskip("bm25s")


def test_bm25_ranks_and_hits(bm25s_installed):
    r = BM25Retriever(chunks())
    hits = r.search("linux install", k=3)
    # c0 has both terms; c2 and c4 one each with equal idf, and the shorter c4 wins
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c4", "c2"]
    assert all(isinstance(h, Hit) for h in hits)
    assert [h.ranks["bm25"] for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score > 0
    assert isinstance(r, Retriever)


def test_bm25_unknown_and_empty_query_return_all_chunks_with_zero_score(bm25s_installed):
    r = BM25Retriever(chunks())
    for q in ("zzzz qqqq", "", "!!!"):
        ranking = r.rank(q)
        assert [i for i, _ in ranking] == list(range(len(DOCS)))  # stable: index order
        assert all(s == 0.0 for _, s in ranking)


def test_bm25_indexes_embed_text_when_set(bm25s_installed):
    cs = chunks()
    cs[1].embed_text = "Guide > Linux\n\n" + cs[1].text
    hits = BM25Retriever(cs).search("guide", k=1)
    assert hits[0].chunk.chunk_id == "c1"


def test_bm25_empty_corpus_and_k(bm25s_installed):
    assert BM25Retriever([]).search("x") == []
    r = BM25Retriever(chunks())
    assert r.search("linux", k=0) == []
    assert len(r.search("linux", k=100)) == len(DOCS)
    with pytest.raises(ValueError):
        r.search("linux", k=-1)


def test_bm25_stemmer_optional(bm25s_installed):
    pytest.importorskip("Stemmer")
    r = BM25Retriever(chunks(), stemmer="english")
    assert r.search("installing", k=1)[0].chunk.chunk_id in ("c0", "c4")


def test_missing_extra_names_the_install_line(monkeypatch):
    monkeypatch.setitem(sys.modules, "bm25s", None)
    with pytest.raises(ImportError, match=r'mdkeychunker\[bm25\]'):
        BM25Retriever(chunks())
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r'mdkeychunker\[dense\]'):
        DenseRetriever(chunks())
    with pytest.raises(ImportError, match=r'mdkeychunker\[rerank\]'):
        Reranker().scores("q", ["d"])


def test_non_chunk_input_rejected():
    with pytest.raises(TypeError):
        DenseRetriever(["text"], embed_fn=bag)  # type: ignore[list-item]


# --------------------------------------------------------------------------- dense

def test_dense_with_embed_fn_and_prefixes():
    seen = []

    def fn(texts):
        seen.extend(texts)
        return bag(texts)

    r = DenseRetriever(chunks(), "nomic", embed_fn=fn)
    assert seen[0].startswith("search_document: ")
    hits = r.search("pricing", k=1)
    assert hits[0].chunk.chunk_id == "c1" and hits[0].ranks == {"dense": 1}
    assert seen[-1] == "search_query: pricing"
    assert 0.0 < hits[0].score <= 1.0 + 1e-6


def test_dense_presets_and_overrides():
    r = DenseRetriever(chunks(), "qwen3-embedding-0.6b", embed_fn=bag)
    assert r.query_prefix.startswith("Instruct: ") and r.query_prefix.endswith("\nQuery:")
    assert r.doc_prefix == ""
    r = DenseRetriever(chunks(), "some/hf-model", embed_fn=bag, query_prefix="q: ")
    assert (r.query_prefix, r.doc_prefix) == ("q: ", "")


def test_dense_batched_search_and_bad_embedder():
    r = DenseRetriever(chunks(), embed_fn=bag)
    res = r.search_many(["pricing", "windows"], k=1)
    assert [h[0].chunk.chunk_id for h in res] == ["c1", "c4"]
    with pytest.raises(ValueError, match="shape"):
        DenseRetriever(chunks(), embed_fn=lambda ts: [[1.0]])
    assert DenseRetriever([], embed_fn=bag).search("x") == []


# --------------------------------------------------------------------------- rerank

def test_rerank_order_ties_and_tail():
    assert rerank_order([5, 6, 7, 8], [0.1, 0.9, 0.1], top=3) == [6, 5, 7, 8]
    assert rerank_order([1, 2], [0.5, 0.5], top=2) == [1, 2]
    with pytest.raises(ValueError):
        rerank_order([1, 2, 3], [0.1], top=3)


def test_qwen3_template_is_the_study_template():
    q, d = format_pair("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "QUERY", "DOC")
    assert q.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert q.endswith("<Instruct>: Given a web search query, retrieve relevant passages that "
                      "answer the query\n<Query>: QUERY\n")
    assert d == "<Document>: DOC<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_reranker_memoises_and_disk_cache(tmp_path):
    calls = []

    def score(pairs):
        calls.append(len(pairs))
        return [float(len(d)) for _, d in pairs]

    r = Reranker("bge-reranker-v2-m3", score_fn=score)
    assert r.scores("q", ["aa", "b", "aa"]) == [2.0, 1.0, 2.0]
    assert r.scores("q", ["b"]) == [1.0] and calls == [2]
    disk = Reranker("bge-reranker-v2-m3", score_fn=score, cache_dir=tmp_path)
    disk.scores("q", ["xyz"])
    again = Reranker("bge-reranker-v2-m3", score_fn=lambda p: pytest.fail("cache miss"),
                     cache_dir=tmp_path)
    assert again.scores("q", ["xyz"]) == [3.0]
    key = (tmp_path / RerankCache(tmp_path, disk.model_key).root.name / "model_key.txt")
    assert key.read_text() == "BAAI/bge-reranker-v2-m3|tmpl=v1|maxlen=2048"


# --------------------------------------------------------------------------- hybrid

def test_hybrid_rrf_matches_manual_fusion(bm25s_installed):
    cs = chunks()
    b = BM25Retriever(cs)
    d = DenseRetriever(cs, embed_fn=bag)
    h = HybridRetriever(cs, bm25=b, dense=d)
    q = "install on linux"
    want = RRF().fuse([b.rank(q), d.rank(q)])
    hits = h.search(q, k=len(cs))
    assert [x.chunk.chunk_id for x in hits] == [f"c{i}" for i, _ in want]
    top = hits[0]
    assert top.ranks["fused"] == 1 and {"bm25", "dense"} <= set(top.ranks)
    assert h.name == "bm25+dense"


def test_hybrid_convex_alpha_extremes_equal_single_retrievers(bm25s_installed):
    cs = chunks()
    b, d = BM25Retriever(cs), DenseRetriever(cs, embed_fn=bag)
    q = "linux service"
    dense_only = HybridRetriever(cs, bm25=b, dense=d, fusion=Convex(alpha=1.0)).rank(q)
    assert [i for i, _ in dense_only][0] == d.rank(q)[0][0]
    sparse_only = HybridRetriever(cs, bm25=b, dense=d, fusion=Convex(alpha=0.0)).rank(q)
    assert [i for i, _ in sparse_only][0] == b.rank(q)[0][0]


def test_hybrid_rerank_reorders_head_only(bm25s_installed):
    cs = chunks()
    d = DenseRetriever(cs, embed_fn=bag)
    rr = Reranker(score_fn=lambda pairs: [float("Windows" in doc) for _, doc in pairs])
    h = HybridRetriever(cs, bm25=False, dense=d, rerank=rr, rerank_depth=3)
    plain = [i for i, _ in d.rank("install")]
    hits = h.search("install", k=len(cs))
    got = [int(x.chunk.chunk_id[1:]) for x in hits]
    head = plain[:3]
    assert sorted(got[:3]) == sorted(head) and got[3:] == plain[3:]
    if 4 in head:
        assert got[0] == 4 and hits[0].ranks["rerank"] == 1 and hits[0].score == 1.0
    assert "rerank" not in hits[-1].ranks
    assert h.name == "dense+rerank"


def test_hybrid_depth_limits_candidates(bm25s_installed):
    cs = chunks()
    h = HybridRetriever(cs, dense=DenseRetriever(cs, embed_fn=bag), depth=2)
    assert len(h.rank("linux")) <= 4


def test_hybrid_validation(bm25s_installed):
    cs = chunks()
    with pytest.raises(ValueError):
        HybridRetriever(cs, bm25=False, dense=None)
    with pytest.raises(ValueError, match="different chunks"):
        HybridRetriever(cs, dense=DenseRetriever(chunks(), embed_fn=bag))
    with pytest.raises(ValueError):
        HybridRetriever(cs, dense=None, rerank_depth=0)


@pytest.mark.slow
def test_dense_with_a_real_sentence_transformer():
    pytest.importorskip("sentence_transformers")
    cs = chunks()
    r = DenseRetriever(cs, "sentence-transformers/all-MiniLM-L6-v2")
    assert r.search("how much does it cost?", k=1)[0].chunk.chunk_id == "c1"
    b = BM25Retriever(cs, method="study")
    h = HybridRetriever(cs, bm25=b, dense=r)
    want = [i for i, _ in RRF().fuse([b.rank("linux install"), r.rank("linux install")])]
    assert [int(x.chunk.chunk_id[1:]) for x in h.search("linux install", k=5)] == want
