"""diagnose on a synthetic corpus where the right answers are known (no model downloads)."""
import json

import pytest

from mdkeychunker.models import Chunk, Hit

pytest.importorskip("numpy")  # the built-in dense retriever needs numpy

from mdkeychunker.diagnose import QA, diagnose  # noqa: E402
from mdkeychunker.evaluate import evidence_recall  # noqa: E402
from mdkeychunker.retrieval import Reranker  # noqa: E402

N_DOCS, N_SECT = 15, 4
BUDGET = 19  # chars/4 tokens = 76 chars: one ~75-char section fits, the next does not


def section(d, s):
    return f"Section {s} of manual {d} explains the topic word zq{d}x{s} in plain detail."


def corpus():
    return [Chunk(text=section(d, s), doc_id=f"d{d}", chunk_id=f"d{d}s{s}",
                  start_char=100 * s, end_char=100 * s + len(section(d, s)))
            for d in range(N_DOCS) for s in range(N_SECT)]


def qa(n_docs=N_DOCS):
    # two questions per document, about sections 1 and 3 (never the first chunk)
    return [{"question": f"what does zq{d}x{s} mean?", "gold": [section(d, s)], "doc_id": f"d{d}"}
            for d in range(n_docs) for s in (1, 3)]


def flat(texts):  # an embedder that cannot tell chunks apart: ranks fall back to index order
    return [[1.0, 0.0] for _ in texts]


def oracle_score(pairs):  # a reranker that knows which chunk holds the query's topic word
    out = []
    for q, d in pairs:
        word = next(w for w in q.replace("?", "").split() if w.startswith("zq"))
        out.append(float(word in d))
    return out


ORACLE = Reranker(score_fn=oracle_score)
KW = dict(bm25="study", embedder=flat, budget_tokens=BUDGET, n_boot=500)


def by_pair(report):
    return {(f.a, f.b): f for f in report.findings}


def test_fusion_check_finds_hybrid_beats_flat_dense_and_reports_mde():
    r = diagnose(corpus(), qa(), check=("fusion",), **KW)
    f = by_pair(r)
    assert f[("hybrid", "dense")].verdict == "better"
    assert f[("hybrid", "dense")].comparison.diff > 0.3
    # RRF with a useless dense half drags the hybrid below BM25 alone: "fusion hurts"
    assert f[("hybrid", "bm25")].verdict == "worse"
    assert "Your fusion hurts" in f[("hybrid", "bm25")].message
    for x in r.findings:
        assert x.comparison.mde == x.comparison.mde  # every verdict carries an MDE
        assert "MDE" in x.message or "Inconclusive" in x.message
    assert r.settings["scope"] == "document" and r.settings["n_clusters"] == N_DOCS
    assert r.arms["bm25"].mean == 1.0 and r.arms["dense"].mean == 0.0


def test_per_question_values_are_evidence_recall_of_the_arm():
    r = diagnose(corpus(), qa(), check=("fusion",), **KW)
    assert sum(len(v) for v in r.per_question["bm25"].values()) == 2 * N_DOCS
    assert r.question_ids["d0"] == ["0", "1"]
    assert all(v == [1.0, 1.0] for v in r.per_question["bm25"].values())


def test_rerank_check_with_an_oracle_reranker():
    r = diagnose(corpus(), qa(), check=("rerank",), compare=[("dense+rerank", "dense")],
                 reranker=ORACLE, **KW)
    f = by_pair(r)
    assert f[("dense+rerank", "dense")].verdict == "better"
    assert ("hybrid+rerank", "hybrid") in f


def test_byo_retrievers_ids_hits_and_document_scope_filtering():
    cs = corpus()

    def mine(query, k):  # returns ids, best first, but includes another document's chunk first
        word = next(w for w in query.replace("?", "").split() if w.startswith("zq"))
        d, s = word[2:].split("x")
        other = f"d{(int(d) + 1) % N_DOCS}s{s}"
        return [other, f"d{d}s{s}"][:k]

    class Bad:
        def search(self, query, k):
            return [Hit(cs[0], 1.0)]

    r = diagnose(cs, qa(), retrievers={"mine": mine, "bad": Bad()}, check=("rerank",),
                 compare=[("mine", "bad")], reranker=ORACLE, **KW)
    assert r.settings["baseline"] == "mine"
    assert r.arms["mine"].mean == 1.0  # the other document's chunk was dropped
    assert by_pair(r)[("mine", "bad")].verdict == "better"
    assert any("baseline" in w for w in r.warnings)


def test_corpus_scope_keeps_other_documents():
    cs = corpus()

    def mine(query, k):
        word = next(w for w in query.replace("?", "").split() if w.startswith("zq"))
        d, s = word[2:].split("x")
        return [f"d{(int(d) + 1) % N_DOCS}s{s}", f"d{d}s{s}"][:k]

    qs = [{k: v for k, v in q.items() if k != "doc_id"} for q in qa()]
    r = diagnose(cs, qs, retrievers={"mine": mine}, check=(), **KW)
    assert r.settings["scope"] == "corpus" and r.settings["cluster_by"] == "question"
    assert r.arms["mine"].mean == 0.0  # the decoy fills the budget


def test_enrich_check_with_enriched_chunks():
    cs = corpus()
    # questions use a synonym that only the "enriched" embed_text contains
    qs = [{"question": f"meaning of term{d}t{s}?", "gold": [section(d, s)], "doc_id": f"d{d}"}
          for d in range(N_DOCS) for s in (1, 3)]
    enriched = [Chunk(text=c.text, doc_id=c.doc_id, chunk_id=c.chunk_id,
                      embed_text=f"Keywords: term{c.doc_id[1:]}t{c.chunk_id[-1]}\n\n{c.text}")
                for c in cs]
    r = diagnose(cs, qs, check=("enrich",), enriched=enriched, **KW)
    f = by_pair(r)
    assert f[("bm25@enriched", "bm25")].verdict == "better"
    assert "enrichment helped most under BM25" in f[("bm25@enriched", "bm25")].context
    with pytest.raises(ValueError, match="enriched"):
        diagnose(cs, qs, check=("enrich",), **KW)
    with pytest.raises(ValueError, match="same documents"):
        diagnose(cs, qs, check=("enrich",), enriched=enriched[:4], **KW)


def test_few_questions_are_inconclusive_with_warnings():
    r = diagnose(corpus(), qa(n_docs=3)[:3], check=("fusion",), **KW)
    assert any("Only 3 questions" in w for w in r.warnings)
    assert any("clusters" in w for w in r.warnings)
    tie = by_pair(r)[("hybrid", "bm25")]
    assert tie.verdict == "inconclusive" and "Inconclusive at n=3" in tie.message


def test_margin_gives_equivalent_and_holm_is_applied():
    from mdkeychunker.retrieval import BM25Retriever

    cs = corpus()
    per_doc = {d: BM25Retriever([c for c in cs if c.doc_id == f"d{d}"], method="study")
               for d in range(N_DOCS)}

    def same_as_bm25(query, k):  # the built-in BM25 arm
        d = int(query.split("zq")[1].split("x")[0])
        return per_doc[d].search(query, k)

    def nearly(query, k):  # BM25, except it misses one question of the 30
        return [] if "zq0x1" in query else same_as_bm25(query, k)

    r = diagnose(cs, qa(), retrievers={"mine": nearly, "same": same_as_bm25},
                 check=("fusion",), compare=[("mine", "bm25"), ("same", "bm25")],
                 margin=0.2, **KW)
    eq = by_pair(r)[("mine", "bm25")]
    assert eq.comparison.diff == pytest.approx(-1 / 30) and eq.verdict == "equivalent"
    assert "Equivalent within +/-20.0 points" in eq.message
    assert by_pair(r)[("same", "bm25")].verdict == "identical"
    assert all(f.p_holm >= f.comparison.p_perm for f in r.findings)
    r2 = diagnose(cs, qa(), retrievers={"mine": nearly}, compare=[("mine", "bm25")],
                  check=(), **KW)
    assert by_pair(r2)[("mine", "bm25")].verdict == "inconclusive"  # no margin, no claim


def test_embedder_check_skipped_when_it_repeats_fusion_and_runs_with_byo():
    r = diagnose(corpus(), qa(), check=("embedder",), upgrade_embedder=flat, **KW)
    assert r.findings == [] and any("skipped" in w for w in r.warnings)

    def mine(query, k):
        return []

    r = diagnose(corpus(), qa(), retrievers={"mine": mine}, check=("embedder",),
                 upgrade_embedder=lambda ts: [[float("zq" in t), 1.0] for t in ts], **KW)
    assert [(f.a, f.b) for f in r.findings] == [("dense-upgrade", "mine")]


def test_markdown_and_dict_outputs():
    r = diagnose(corpus(), qa(), check=("fusion", "rerank"), reranker=ORACLE, **KW)
    md = r.to_markdown()
    for part in ("# Retrieval diagnosis", "## Verdicts", "## Retrievers", "| MDE |",
                 "## Context from the v4 study", "Study scope"):
        assert part in md
    assert "42.7 vs 47.8" in md  # the study's fusion finding, as context
    json.dumps(r.to_dict())
    assert r.verdicts == [f.message for f in r.findings]


def test_markdown_input_is_chunked_with_mk_chunk():
    # 8 sections -> 4 chunks per document (mk.chunk merges short sections in pairs,
    # min_chars=100); with only 2 chunks, BM25 idf of a term in one of them is ln(1.5/1.5) = 0
    docs = {f"d{d}": "\n\n".join(f"# S{s}\n\n{section(d, s)}" for s in range(8))
            for d in range(N_DOCS)}
    r = diagnose(docs, qa(), check=("fusion",), **{**KW, "budget_tokens": 45})
    assert r.arms["bm25"].mean == 1.0


def test_input_validation():
    cs = corpus()
    with pytest.raises(ValueError, match="unknown check"):
        diagnose(cs, qa(), check=("vibes",), **KW)
    with pytest.raises(ValueError, match="names"):
        diagnose(cs, qa(), retrievers={"bm25": lambda q, k: []}, check=(), **KW)
    with pytest.raises(ValueError, match="doc_id"):
        diagnose(cs, [{"question": "x", "gold": ["y"]}], scope="document", **KW)
    with pytest.raises(ValueError, match="no chunks"):
        diagnose(cs, [{"question": "x", "gold": ["y"], "doc_id": "zzz"}], **KW)
    with pytest.raises(ValueError):
        diagnose(cs, [], **KW)
    with pytest.raises(ValueError, match="nothing to do"):
        diagnose(cs, qa(), check=(), **KW)
    with pytest.raises(TypeError):
        diagnose([3], qa(), **KW)
    q = QA("what?", [], "d0")
    with pytest.raises(ValueError, match="no gold"):
        diagnose(cs, [q], **KW)


def test_qa_forms():
    cs = corpus()
    forms = [("what does zq0x1 mean?", section(0, 1), "d0"),
             {"query": "what does zq0x3 mean?", "evidence": [section(0, 3)], "doc_id": "d0"}]
    r = diagnose(cs, forms, check=("fusion",), **KW)
    assert r.settings["n_questions"] == 2


def test_offset_gold_spans_work_with_offset_chunks():
    cs = corpus()
    qs = [{"question": f"what does zq{d}x1 mean?", "gold": [(f"d{d}", 100, 100 + 20)],
           "doc_id": f"d{d}"} for d in range(N_DOCS)]
    r = diagnose(cs, qs, check=("fusion",), **KW)
    assert r.arms["bm25"].mean == 1.0
    assert evidence_recall([cs[1]], [("d0", 100, 120)]) == 1.0


# --------------------------------------------------------------------------- degenerate data

def toy():
    """The lead's new-user case: 6 short documents, 18 questions, default 512-token budget."""
    docs = {f"t{d}": "\n\n".join(f"# Part {s}\n\nPart {s} of note {d} is about yy{d}w{s}."
                                 for s in range(3)) for d in range(6)}
    qs = [{"question": f"what is yy{d}w{s}?", "gold": [f"Part {s} of note {d} is about yy{d}w{s}."],
           "doc_id": f"t{d}"} for d in range(6) for s in range(3)]
    return docs, qs


def test_budget_covering_whole_documents_is_flagged_and_identical_arms_say_so():
    docs, qs = toy()
    r = diagnose(docs, qs, check=("fusion",), bm25="study", embedder=flat, n_boot=300)
    assert any("Your budget_tokens (512) covers whole documents for 18 of 18 questions" in w
               and "scope='corpus'" in w for w in r.warnings)
    assert all(a.mean == 1.0 for a in r.arms.values())
    for f in r.findings:
        assert f.verdict == "identical"
        assert "No difference: identical on every question" in f.message
        assert "0.0 points cannot be detected" not in f.message
    md = r.to_markdown()
    assert "identical" in md and "covers whole documents" in md
    # a budget smaller than a document removes the warning
    r2 = diagnose(docs, qs, check=("fusion",), bm25="study", embedder=flat, n_boot=300,
                  budget_tokens=8)
    assert not any("covers whole documents" in w for w in r2.warnings)


def test_zero_variation_prints_mde_as_not_available():
    cs = corpus()

    def perfect(query, k):
        word = next(w for w in query.replace("?", "").split() if w.startswith("zq"))
        d, s = word[2:].split("x")
        return [f"d{d}s{s}"]

    r = diagnose(cs, qa(), retrievers={"perfect": perfect, "empty": lambda q, k: []},
                 compare=[("perfect", "empty")], check=(), **KW)
    f = by_pair(r)[("perfect", "empty")]
    assert f.comparison.mde == 0.0
    assert "MDE n/a (no variation)" in f.message
    assert "| n/a (no variation) |" in r.to_markdown()


def test_missing_extra_is_reported_with_the_check_that_needs_it(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("sentence_transformers", "bm25s")
                        else real(name, *a))
    with pytest.raises(ImportError, match=r"check='fusion' needs a dense embedder.*embedder="):
        diagnose(corpus(), qa(), check=("fusion",), bm25="study", budget_tokens=BUDGET)
    with pytest.raises(ImportError, match=r"check='fusion' needs BM25.*bm25='study'"):
        diagnose(corpus(), qa(), check=("fusion",), embedder=flat, budget_tokens=BUDGET)
    with pytest.raises(ImportError, match=r"check='rerank' needs a reranker"):
        diagnose(corpus(), qa(), check=("rerank",), **KW)
    # a BYO reranker / embedder needs nothing
    diagnose(corpus(), qa(), check=("rerank",), reranker=ORACLE, **KW)


def test_enrich_is_in_the_default_checks_only_when_enriched_chunks_are_given():
    cs = corpus()
    enriched = [Chunk(text=c.text, doc_id=c.doc_id, chunk_id=c.chunk_id, embed_text=c.text)
                for c in cs]
    r = diagnose(cs, qa(), reranker=ORACLE, upgrade_embedder=flat, enriched=enriched, **KW)
    assert {f.check for f in r.findings} == {"fusion", "rerank", "enrich"}
    r = diagnose(cs, qa(), reranker=ORACLE, upgrade_embedder=flat, **KW)
    assert {f.check for f in r.findings} == {"fusion", "rerank"}


# --------------------------------------------------------------------------- embedder memory knobs

def capture_load_embedder(monkeypatch):
    import mdkeychunker.retrieval.dense as dense

    calls, seen = [], []

    def fake(model, **kwargs):
        calls.append((model, kwargs))

        def embed(texts):
            seen.extend(texts)
            return flat(texts)
        embed.model = model
        return embed

    monkeypatch.setattr(dense, "load_embedder", fake)
    return calls, seen


def test_embed_batch_size_and_max_seq_length_reach_load_embedder(monkeypatch):
    calls, seen = capture_load_embedder(monkeypatch)
    diagnose(corpus(), qa(), check=("fusion",), bm25="study", embedder="nomic",
             budget_tokens=BUDGET, n_boot=200, embed_batch_size=4, max_seq_length=2048)
    assert calls == [("nomic", {"batch_size": 4, "max_seq_length": 2048})]
    assert any(t.startswith("search_document: ") for t in seen)  # the preset's prefixes
    calls.clear()
    diagnose(corpus(), qa(), check=("fusion",), bm25="study", embedder="nomic",
             budget_tokens=BUDGET, n_boot=200)
    assert calls == [("nomic", {})]  # defaults unchanged


def test_a_load_embedder_result_keeps_its_preset_prefixes(monkeypatch):
    calls, seen = capture_load_embedder(monkeypatch)
    import mdkeychunker.retrieval.dense as dense

    fn = dense.load_embedder("nomic", batch_size=4)
    calls.clear()
    r = diagnose(corpus(), qa(), check=("fusion",), bm25="study", embedder=fn,
                 budget_tokens=BUDGET, n_boot=200)
    assert calls == []  # used as given, not reloaded
    assert any(t.startswith("search_query: what does") for t in seen)
    assert r.settings["embedder"] == "nomic"
