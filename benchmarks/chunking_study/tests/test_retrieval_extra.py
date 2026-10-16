"""Offline tests for retrieval_extra / evaluate_extra (tiny fake models; no network, no GPU).

Run:  .venv-ml/bin/python -m pytest tests/test_retrieval_extra.py -q
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import evaluate  # noqa: E402
import evaluate_v2  # noqa: E402
import retrieval_extra as rx  # noqa: E402


# --------------------------------------------------------------------------- fakes

class CharEncoder:
    """One token per non-space word; offsets are exact character spans.

    Token embedding = one-hot(word hash) + window_tag * e_last, where window_tag is
    the window's first content id (lets tests see which window was used)."""

    def __init__(self, max_len: int = 64, prefix_len: int = 2, dim: int = 16, context: bool = False):
        self.max_len, self.prefix_len, self.dim, self.context = max_len, prefix_len, dim, context
        self.calls: list[list[int]] = []
        self.vocab: dict[str, int] = {}

    def tokenize(self, text):
        import re
        ids, offs = [], []
        for m in re.finditer(r"\S+", text):
            ids.append(self.vocab.setdefault(m.group(), len(self.vocab) + 10))
            offs.append((m.start(), m.end()))
        return ids, offs

    def content_capacity(self, task):
        return self.max_len - 2 - self.prefix_len

    def wrap(self, content_ids, task):
        return [1] + [2] * self.prefix_len + list(content_ids) + [3], 1 + self.prefix_len

    def token_embeddings(self, input_ids, task):
        self.calls.append(list(input_ids))
        E = np.zeros((len(input_ids), self.dim), dtype=np.float32)
        for i, t in enumerate(input_ids):
            E[i, t % (self.dim - 1)] = 1.0
            if self.context:
                E[i, self.dim - 1] = len(self.calls)  # which window call produced it
        return E


class TestRRF(unittest.TestCase):
    def test_matches_evaluate_v2(self):
        a, b = [3, 1, 2, 0], [0, 2, 1, 3]
        self.assertEqual(rx.rrf(a, b), evaluate_v2.rrf(a, b))

    def test_scores(self):
        # item 1: 1/61 + 1/62 beats item 0: 1/62 + 1/63 ... and ties keep first-seen order
        self.assertEqual(rx.rrf([1, 0], [1, 0]), [1, 0])
        self.assertEqual(rx.rrf([0, 1], [1, 0]), [0, 1])  # equal scores -> insertion order
        self.assertEqual(rx.rrf([5], [], k=60), [5])

    def test_dense_orders(self):
        D = rx._norm(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
        Q = rx._norm(np.array([[0, 1]], dtype=np.float32))
        self.assertEqual(rx.dense_orders(D, Q), [[1, 2, 0]])


class TestRerank(unittest.TestCase):
    def test_rerank_order_head_only(self):
        order = [4, 3, 2, 1, 0]
        self.assertEqual(rx.rerank_order(order, [0.1, 0.9, 0.5], top=3), [3, 2, 4, 1, 0])

    def test_ties_keep_first_stage(self):
        self.assertEqual(rx.rerank_order([7, 8, 9], [1.0, 1.0, 2.0], top=3), [9, 7, 8])

    def test_reranker_with_fake_and_cache(self):
        calls = []

        def fake(pairs):
            calls.append(len(pairs))
            return [float(len(d)) for _, d in pairs]  # longer doc = more relevant

        with tempfile.TemporaryDirectory() as td:
            rr = rx.Reranker("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", cache_root=Path(td), score_fn=fake)
            docs = ["a", "bbb", "cc", "dddd"]
            self.assertEqual(rr.rerank([0, 1, 2, 3], "q", docs, top=3), [1, 2, 0, 3])
            self.assertEqual(calls, [3])
            # second call: all from cache (memory and disk)
            rr2 = rx.Reranker("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", cache_root=Path(td), score_fn=fake)
            self.assertEqual(rr2.rerank([0, 1, 2, 3], "q", docs, top=3), [1, 2, 0, 3])
            self.assertEqual(calls, [3])
            self.assertEqual(rr2.cache_hits, 3)
            # a new doc only scores the new pair; duplicates are scored once
            rr2.scores("q", ["dddd", "dddd", "a"])
            self.assertEqual(calls, [3, 1])
            # a different max_length is a different cache namespace
            rr3 = rx.Reranker("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", max_length=512, cache_root=Path(td),
                              score_fn=fake)
            rr3.scores("q", ["a"])
            self.assertEqual(calls, [3, 1, 1])

    def test_cache_keys(self):
        self.assertNotEqual(rx.RerankCache.query_hash("x"), rx.RerankCache.text_hash("x"))
        self.assertEqual(rx.RerankCache.text_hash("abc"), rx.RerankCache.text_hash("abc"))
        a = rx.Reranker("m1", cache_root=Path("/nonexistent"), score_fn=lambda p: [0] * len(p))
        b = rx.Reranker("m2", cache_root=Path("/nonexistent"), score_fn=lambda p: [0] * len(p))
        self.assertNotEqual(a.cache.root, b.cache.root)

    def test_qwen_template(self):
        q, d = rx.format_pair(rx.QWEN3_RERANKER, "what?", "doc text")
        self.assertTrue(q.startswith("<|im_start|>system\n") and q.endswith("<Query>: what?\n"))
        self.assertIn("<Instruct>: " + rx.RERANK_INSTRUCTION, q)
        self.assertEqual(d, "<Document>: doc text<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(rx.format_pair(rx.BGE_RERANKER, "q", "d"), ("q", "d"))


class TestQwen3Embedding(unittest.TestCase):
    def test_register_and_prefix(self):
        old = dict(evaluate.EMBEDDERS), evaluate.OLLAMA
        try:
            rx.register_qwen3("http://127.0.0.1:9")
            model, qpre, dpre = evaluate.EMBEDDERS["qwen3"]
            self.assertEqual(model, "qwen3-embedding:0.6b")
            self.assertEqual(qpre, "Instruct: Given a web search query, retrieve relevant passages that answer "
                                   "the query\nQuery:")
            self.assertEqual(dpre, "")
            self.assertEqual(evaluate.OLLAMA, "http://127.0.0.1:9")
        finally:
            evaluate.EMBEDDERS.clear()
            evaluate.EMBEDDERS.update(old[0])
            evaluate.OLLAMA = old[1]


class TestSpans(unittest.TestCase):
    def test_exact_and_duplicates(self):
        md = "# T\n\nsame\n\nmid\n\nsame\n"
        spans, c = rx.locate_spans(md, ["# T", "same", "mid", "same"])
        self.assertEqual(spans, [(0, 3), (5, 9), (11, 14), (16, 20)])
        self.assertEqual(c["exact_cursor"], 4)

    def test_whitespace_and_missing(self):
        md = "alpha  beta\ngamma delta"
        spans, c = rx.locate_spans(md, ["alpha beta gamma", "nothere at all", "  "])
        self.assertEqual(spans[0], (0, 17))
        self.assertIsNone(spans[1])
        self.assertIsNone(spans[2])
        self.assertEqual((c["whitespace_normalised"], c["not_found"], c["empty"]), (1, 1, 1))

    def test_token_range(self):
        offs = [(0, 2), (3, 5), (5, 5), (6, 9)]
        self.assertEqual(rx.token_range(offs, (3, 9)), (1, 3))
        self.assertEqual(rx.token_range(offs, (4, 5)), (1, 1))
        self.assertIsNone(rx.token_range(offs, (9, 12)))


class TestWindows(unittest.TestCase):
    def test_macro_windows(self):
        self.assertEqual(rx.macro_windows(5, 10, 3), [(0, 5)])
        w = rx.macro_windows(25, 10, 4)
        self.assertEqual(w, [(0, 10), (6, 16), (12, 22), (15, 25)])
        covered = set().union(*(range(s, e) for s, e in w))
        self.assertEqual(covered, set(range(25)))

    def test_best_window_most_context(self):
        w = [(0, 10), (6, 16), (12, 22)]
        self.assertEqual(rx.best_window((7, 8), w), 0)    # w0 min(7,1)=1, w1 min(1,7)=1 -> tie, first
        self.assertEqual(rx.best_window((8, 8), w), 1)    # w0 min(8,1)=1, w1 min(2,7)=2
        self.assertEqual(rx.best_window((10, 11), w), 1)  # only w1 contains it
        self.assertIsNone(rx.best_window((0, 20), w))
        self.assertEqual(rx.best_window((13, 14), [(0, 20), (10, 30)]), 0)  # min(13,5)=5 vs min(3,15)=3

    def test_token_best_window(self):
        self.assertEqual(rx.token_best_window(9, [(0, 10), (6, 16)]), 1)


class TestLateChunking(unittest.TestCase):
    def test_pooling_single_window(self):
        md = "aa bb cc\n\ndd ee"
        enc = CharEncoder(max_len=64)
        E, rep = rx.late_chunk(enc, md, ["aa bb cc", "dd ee"])
        ids, _ = enc.tokenize(md)
        full = enc.token_embeddings(enc.wrap(ids, "p")[0], "p")
        first = 1 + enc.prefix_len
        exp0 = rx._norm(full[first:first + 3].mean(0))
        exp1 = rx._norm(full[first + 3:first + 5].mean(0))
        np.testing.assert_allclose(E[0], exp0, atol=1e-6)
        np.testing.assert_allclose(E[1], exp1, atol=1e-6)
        self.assertEqual((rep.n_windows, rep.pooled, rep.fallback_plain, rep.split_across_windows), (1, 2, 0, 0))
        self.assertEqual(len(enc.calls), 2)  # one window forward (+ the one made above)

    def test_prefix_and_special_tokens_excluded(self):
        md = "x y"
        enc = CharEncoder(max_len=64, prefix_len=3)
        E, _ = rx.late_chunk(enc, md, ["x y"])
        ids, _ = enc.tokenize(md)
        v = np.zeros(enc.dim, dtype=np.float32)
        for t in ids:
            v[t % (enc.dim - 1)] += 0.5
        np.testing.assert_allclose(E[0], rx._norm(v), atol=1e-6)

    def test_macro_windows_choose_most_context(self):
        words = [f"w{i}" for i in range(30)]
        md = " ".join(words)
        enc = CharEncoder(max_len=14, prefix_len=2, context=True)  # capacity 10
        chunks = [" ".join(words[0:3]), " ".join(words[11:13]), " ".join(words[27:30])]
        E, rep = rx.late_chunk(enc, md, chunks, overlap=4)
        # windows: (0,10) (6,16) (12,22) (18,28) (20,30)
        self.assertEqual(rep.n_windows, 5)
        # windows actually run: 0, 1 and 4, in that order
        self.assertEqual(len(enc.calls), 3)
        self.assertEqual(enc.calls[2][3], enc.tokenize(md)[0][20])
        # chunk 1 (tokens 11-12) must come from window 1 (6,16): ctx min(5,3)=3 vs window 2 not containing 11
        self.assertEqual(enc.calls[1][3], enc.tokenize(md)[0][6])
        self.assertEqual(rep.split_across_windows, 0)

    def test_split_and_fallback(self):
        words = [f"w{i}" for i in range(20)]
        md = " ".join(words)
        enc = CharEncoder(max_len=10, prefix_len=0)  # capacity 8
        long_chunk = " ".join(words[2:18])  # 16 tokens > capacity
        plain_calls = []

        def plain(texts):
            plain_calls.append(texts)
            return np.ones((len(texts), enc.dim), dtype=np.float32)

        E, rep = rx.late_chunk(enc, md, [long_chunk, "missing text here"], overlap=4, plain_fn=plain)
        self.assertEqual(rep.split_across_windows, 1)
        self.assertEqual(rep.fallback_plain, 1)
        self.assertEqual(plain_calls, [["missing text here"]])
        np.testing.assert_allclose(E[1], rx._norm(np.ones(enc.dim)), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(E[0])), 1.0, places=5)
        with self.assertRaises(ValueError):
            rx.late_chunk(enc, md, ["missing"])

    def test_lc_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            enc = CharEncoder(max_len=64)
            E1, r1 = rx.late_chunk_cached(enc, "doc", "aa bb", ["aa", "bb"], root=Path(td))
            n = len(enc.calls)
            E2, r2 = rx.late_chunk_cached(enc, "doc", "aa bb", ["aa", "bb"], root=Path(td))
            self.assertEqual(len(enc.calls), n)
            np.testing.assert_allclose(E1, E2)
            self.assertEqual(r1.as_dict(), r2.as_dict())
            rx.late_chunk_cached(enc, "doc", "aa bb", ["aa bb"], root=Path(td))  # new chunking -> new key
            self.assertEqual(len(enc.calls), n + 1)

    def test_cached_encode(self):
        with tempfile.TemporaryDirectory() as td:
            seen = []

            def enc(texts, task):
                seen.extend(texts)
                return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

            A = rx.cached_encode(["a", "bbb"], "retrieval.passage", enc, root=Path(td))
            B = rx.cached_encode(["bbb", "a", "cc"], "retrieval.passage", enc, root=Path(td))
            self.assertEqual(sorted(seen), ["a", "bbb", "cc"])
            np.testing.assert_allclose(A[0], B[1])
            rx.cached_encode(["a"], "retrieval.query", enc, root=Path(td))  # task is part of the key
            self.assertEqual(seen.count("a"), 2)


class TestEvaluateExtra(unittest.TestCase):
    def test_orders_and_stats_end_to_end(self):
        import evaluate_extra as ex
        docs = ["the cat sat on the mat", "dogs bark loudly at night", "cats like warm mats"]
        queries = ["where did the cat sit", "what do dogs do"]
        rr = rx.Reranker("fake", cache_root=Path(tempfile.mkdtemp()),
                         score_fn=lambda pairs: [float("dog" in d) for _, d in pairs])

        def fake_rank(r, d, q):
            return [list(range(len(d)))[::-1] for _ in q]

        arms = ex.Arms({"rerank"}, rr, None, log=lambda s: None)
        old = ex.ev.rank_all
        try:
            ex.ev.rank_all = lambda d, q, rs: {r: fake_rank(r, d, q) for r in rs}
            o = arms.orders("S", docs, queries)
        finally:
            ex.ev.rank_all = old
        self.assertEqual(o["hybrid"][0], [2, 1, 0])
        self.assertEqual(o["hybrid+rerank"][0], [1, 2, 0])  # dog doc first, ties keep hybrid order

        per = {("hybrid", "S", "rec@512t"): {"p1": [0.5, 1.0], "p2": [0.0]},
               ("hybrid", "LC-S", "rec@512t"): {"p1": [1.0, 1.0], "p2": [0.0]},
               ("hybrid+rerank", "S", "rec@512t"): {"p1": [1.0, 1.0], "p2": [1.0]}}
        rows = ex.secondary_rows({"qasper": {"per": per, "question_ids": {}, "strata": None}}, n=200)
        kinds = {(r["retriever"], r["a"], r["b"]) for r in rows}
        self.assertIn(("hybrid", "LC-S", "S"), kinds)
        self.assertIn(("hybrid+rerank vs hybrid", "S", "S"), kinds)
        self.assertTrue(all(r["adjustment"] == "none" for r in rows))
        cross = next(r for r in rows if r["retriever"] == "hybrid+rerank vs hybrid")
        self.assertAlmostEqual(cross["diff"], (0.5 + 0 + 1) / 3)

    def test_merge_dump(self):
        import evaluate_extra as ex
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "qasper_per_question.json"
            qids = {"p1": ["q1"]}
            ex.merge_dump(p, {"per": {("qwen3", "S", "m"): {"p1": [1.0]}}, "question_ids": qids}, {}, True)
            ex.merge_dump(p, {"per": {("hybrid+rerank", "S", "m"): {"p1": [0.0]}}, "question_ids": qids}, {}, True)
            per = evaluate_v2.load_per_question(p)["per"]
            self.assertEqual(set(per), {("qwen3", "S", "m"), ("hybrid+rerank", "S", "m")})
            msgs = []
            ex.merge_dump(p, {"per": {("jina", "S", "m"): {"p2": [0.0]}}, "question_ids": {"p2": ["q"]}}, {}, True,
                          log=msgs.append)
            self.assertEqual(set(evaluate_v2.load_per_question(p)["per"]), {("jina", "S", "m")})
            self.assertTrue(msgs and "not merging" in msgs[0])


if __name__ == "__main__":
    unittest.main()
