"""Offline tests for llm_api, answer, judge and stats (no network; HTTP is mocked).

Run:  .venv/bin/python -m unittest tests/test_harness_v2.py -v
      (or .venv/bin/python tests/test_harness_v2.py; pytest also collects it)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import answer  # noqa: E402
import llm_api  # noqa: E402
import stats  # noqa: E402

HAS_JUDGE = importlib.util.find_spec("judge") is not None
FAKE_KEY = "sk-or-v1-TESTKEY000000000000000000"


# --------------------------------------------------------------------------- fakes

class FakeResp:
    def __init__(self, status: int, body, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeHTTP:
    """Returns queued responses (or raises queued exceptions); records requests."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def post(self, url, json=None, headers=None, timeout=None):
        with self.lock:
            self.calls.append({"url": url, "json": json, "headers": headers})
            item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def or_ok(text="OK", cost=0.0001, provider="DeepInfra", reasoning_tokens=5):
    return FakeResp(200, {
        "id": "gen-1", "model": "openai/gpt-oss-120b", "provider": provider,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "cost": cost,
                  "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}})


class ClientCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.tmp.name) / "ledger.jsonl"
        self.sleeps: list[float] = []

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, http, cap=5.5, retries=5):
        return llm_api.LLMClient(ledger=llm_api.Ledger(self.ledger_path), budget_usd=cap, max_retries=retries,
                                 http=http, sleep=self.sleeps.append, secret=lambda name: FAKE_KEY,
                                 gemini_key=lambda: "AIzaFAKEFAKEFAKEFAKEFAKEFAKE0000")

    def rows(self):
        return [json.loads(x) for x in self.ledger_path.read_text().splitlines()]


# --------------------------------------------------------------------------- llm_api

class TestSpecs(unittest.TestCase):
    def test_parse(self):
        s = llm_api.ModelSpec.parse("openrouter:openai/gpt-oss-120b@deepinfra/bf16,deepinfra/turbo")
        self.assertEqual((s.backend, s.model, s.providers),
                         ("openrouter", "openai/gpt-oss-120b", ("deepinfra/bf16", "deepinfra/turbo")))
        self.assertEqual(llm_api.ModelSpec.parse("ollama:qwen2.5:7b").model, "qwen2.5:7b")
        with self.assertRaises(ValueError):
            llm_api.ModelSpec.parse("foo:bar")

    def test_family(self):
        f = llm_api.model_family
        self.assertEqual(f("ollama:qwen2.5:7b"), "qwen")
        self.assertEqual(f("openrouter:meta-llama/llama-3.3-70b-instruct@deepinfra/turbo"), "llama")
        self.assertEqual(f("openrouter:openai/gpt-oss-120b@deepinfra/bf16"), "gpt-oss")
        self.assertEqual(f("gemini:gemini-3.1-flash-lite"), "gemini")

    def test_extract_json(self):
        self.assertEqual(llm_api.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm_api.extract_json('Sure: {"a": [1,2]} done'), {"a": [1, 2]})
        self.assertIsNone(llm_api.extract_json("no json here"))


class TestOpenRouter(ClientCase):
    def test_request_shape_and_result(self):
        http = FakeHTTP(or_ok('{"x": 1}'))
        r = self.client(http).complete("openrouter:openai/gpt-oss-120b@deepinfra/bf16", "hello",
                                       json_mode=True, max_tokens=99, temperature=0, seed=7, reasoning="low")
        body = http.calls[0]["json"]
        self.assertEqual(body["provider"]["order"], ["deepinfra/bf16"])
        self.assertIs(body["provider"]["allow_fallbacks"], False)
        self.assertEqual(body["provider"]["data_collection"], "deny")
        self.assertIn("together", body["provider"]["ignore"])
        self.assertEqual(body["usage"], {"include": True})
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual((body["seed"], body["max_tokens"], body["temperature"]), (7, 99, 0))
        for k in ("text", "parsed_json", "in_tokens", "out_tokens", "cost_usd", "provider", "latency_s", "model"):
            self.assertIn(k, r)
        self.assertEqual(r["parsed_json"], {"x": 1})
        self.assertEqual((r["provider"], r["cost_usd"], r["reasoning_tokens"]), ("DeepInfra", 0.0001, 5))

    def test_reasoning_variants(self):
        f = llm_api._openrouter_reasoning
        self.assertEqual(f("off"), {"enabled": False})
        self.assertEqual(f("exclude"), {"exclude": True})
        self.assertIsNone(f(None))
        with self.assertRaises(ValueError):
            f("max")

    def test_unpinned_and_banned_rejected(self):
        c = self.client(FakeHTTP(or_ok()))
        with self.assertRaises(llm_api.LLMError):
            c.complete("openrouter:openai/gpt-oss-120b", "x")
        with self.assertRaises(llm_api.LLMError):
            c.complete("openrouter:openai/gpt-oss-120b@together", "x")

    def test_retry_429_then_ok_honours_retry_after(self):
        http = FakeHTTP(FakeResp(429, {"error": {"message": "rate"}}, {"retry-after": "7"}), or_ok())
        r = self.client(http).complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(r["attempts"], 2)
        self.assertGreaterEqual(self.sleeps[0], 7)

    def test_retry_5xx_and_timeout(self):
        import requests
        http = FakeHTTP(FakeResp(503, "busy"), requests.exceptions.ReadTimeout("t"), or_ok())
        r = self.client(http).complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(r["attempts"], 3)
        self.assertEqual(len(self.sleeps), 2)
        self.assertLess(self.sleeps[0], self.sleeps[1])  # exponential backoff

    def test_in_body_200_error_retried_and_logged(self):
        wrapped = FakeResp(200, {"error": {"code": 502, "message": "Upstream error",
                                           "metadata": {"provider_name": "DeepInfra"}},
                                 "usage": {"cost": 0.00002}})
        http = FakeHTTP(wrapped, or_ok(cost=0.0001))
        c = self.client(http)
        r = c.complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(r["attempts"], 2)
        rows = self.rows()
        self.assertEqual([x["status"] for x in rows], ["retry", "ok"])
        self.assertAlmostEqual(c.ledger.total_spend(), 0.00012)  # billed error counted

    def test_in_body_4xx_not_retried(self):
        http = FakeHTTP(FakeResp(200, {"error": {"code": 400, "message": "bad param"}}), or_ok())
        with self.assertRaises(llm_api.LLMError):
            self.client(http).complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(self.rows()[-1]["status"], "error")

    def test_http_400_not_retried(self):
        http = FakeHTTP(FakeResp(400, {"error": {"message": "no"}}))
        with self.assertRaises(llm_api.LLMError):
            self.client(http).complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(len(http.calls), 1)

    def test_gives_up_after_max_retries(self):
        http = FakeHTTP(FakeResp(502, "bad gateway"))
        with self.assertRaises(llm_api.LLMError):
            self.client(http, retries=3).complete("openrouter:m/x@deepinfra", "p")
        self.assertEqual(len(http.calls), 4)

    def test_budget_guard_reads_ledger(self):
        self.ledger_path.write_text("".join(json.dumps({"cost_usd": 0.4}) + "\n" for _ in range(3)))
        c = self.client(FakeHTTP(or_ok(cost=0.01)), cap=1.204)  # 1.2 spent + 0.005 reserve > cap
        with self.assertRaises(llm_api.BudgetExceeded):
            c.complete("openrouter:m/x@deepinfra", "p")
        c2 = self.client(FakeHTTP(or_ok(cost=0.01)), cap=1.3)
        c2.complete("openrouter:m/x@deepinfra", "p")
        self.assertAlmostEqual(c2.ledger.total_spend(), 1.21)

    def test_budget_guard_stops_mid_run_and_free_backends_unaffected(self):
        c = self.client(FakeHTTP(or_ok(cost=0.023)), cap=0.05)
        c.complete("openrouter:m/x@deepinfra", "p")
        c.complete("openrouter:m/x@deepinfra", "p")
        with self.assertRaises(llm_api.BudgetExceeded):
            c.complete("openrouter:m/x@deepinfra", "p")  # 0.046 spent + 0.005 reserve > 0.05
        ollama = FakeHTTP(FakeResp(200, {"message": {"content": "hi"}, "prompt_eval_count": 3, "eval_count": 1}))
        c.http = ollama
        self.assertEqual(c.complete("ollama:qwen2.5:7b", "p")["text"], "hi")

    def test_ledger_rows_have_no_prompt_or_key(self):
        c = self.client(FakeHTTP(or_ok()))
        c.complete("openrouter:m/x@deepinfra", "SECRET PROMPT TEXT")
        raw = self.ledger_path.read_text()
        self.assertNotIn("SECRET PROMPT TEXT", raw)
        self.assertNotIn(FAKE_KEY, raw)
        row = self.rows()[0]
        for k in ("ts", "model", "provider", "prompt_sha256", "in_tokens", "out_tokens", "cost_usd"):
            self.assertIn(k, row)

    def test_errors_are_redacted(self):
        http = FakeHTTP(FakeResp(401, {"error": {"message": f"bad key {FAKE_KEY}"}}))
        with self.assertRaises(llm_api.LLMError) as cm:
            self.client(http).complete("openrouter:m/x@deepinfra", "p")
        self.assertNotIn(FAKE_KEY, str(cm.exception))
        self.assertNotIn(FAKE_KEY, self.ledger_path.read_text())

    def test_thread_safe_ledger_and_budget(self):
        c = self.client(FakeHTTP(or_ok(cost=0.001)), cap=10)
        errs = []

        def work():
            try:
                for _ in range(10):
                    c.complete("openrouter:m/x@deepinfra", "p")
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=work) for _ in range(16)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertFalse(errs)
        rows = self.rows()  # every line parses: no torn writes
        self.assertEqual(len(rows), 160)
        self.assertAlmostEqual(llm_api.Ledger(self.ledger_path).total_spend(), 0.16)

    def test_concurrent_calls_cannot_overshoot_cap(self):
        # reserve 0.005/call, cap 0.02: at most 4 in flight before any ledger row lands
        gate = threading.Event()

        class SlowHTTP(FakeHTTP):
            def post(self, *a, **k):
                gate.wait(2)
                return super().post(*a, **k)

        c = self.client(SlowHTTP(or_ok(cost=0.005)), cap=0.02)
        res = []

        def one():
            try:
                c.complete("openrouter:m/x@deepinfra", "p")
                res.append("ok")
            except llm_api.BudgetExceeded:
                res.append("refused")

        ts = [threading.Thread(target=one) for _ in range(8)]
        [t.start() for t in ts]
        gate.set()
        [t.join() for t in ts]
        self.assertLessEqual(res.count("ok"), 4)
        self.assertLessEqual(c.ledger.total_spend(), 0.02 + 1e-9)


class TestOtherBackends(ClientCase):
    def test_ollama_body(self):
        http = FakeHTTP(FakeResp(200, {"message": {"content": '{"a":1}'}, "prompt_eval_count": 10,
                                       "eval_count": 4, "model": "qwen2.5:7b"}))
        r = self.client(http).complete("ollama:qwen2.5:7b", "p", json_mode=True, max_tokens=33, seed=0)
        body = http.calls[0]["json"]
        self.assertIs(body["think"], False)
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["options"]["num_ctx"], llm_api.NUM_CTX)
        self.assertEqual((body["options"]["seed"], body["options"]["num_predict"]), (0, 33))
        self.assertEqual((r["in_tokens"], r["out_tokens"], r["cost_usd"], r["parsed_json"]), (10, 4, 0.0, {"a": 1}))

    def test_gemini_key_in_header_not_url(self):
        http = FakeHTTP(FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]},
                                                       "finishReason": "STOP"}],
                                       "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}}))
        r = self.client(http).complete("gemini:gemini-3.1-flash-lite", "p", reasoning="minimal")
        call = http.calls[0]
        self.assertNotIn("key=", call["url"])
        self.assertIn("x-goog-api-key", call["headers"])
        self.assertEqual(call["json"]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertEqual((r["text"], r["cost_usd"]), ("OK", 0.0))

    def test_gemini_429_retried_without_rotation(self):
        ok = FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}], "usageMetadata": {}})
        http = FakeHTTP(FakeResp(429, {"error": {"message": "quota"}}), ok)
        self.client(http).complete("gemini:gemini-2.5-flash-lite", "p")
        keys = {c["headers"]["x-goog-api-key"] for c in http.calls}
        self.assertEqual(len(keys), 1)


# --------------------------------------------------------------------------- answer

class TestQasperF1(unittest.TestCase):
    # Expected values computed with the official evaluator
    # (allenai/qasper-led-baseline scripts/evaluator.py, fetched 2026-10-06).
    CASES = [("the cat sat", "a cat sat down", 0.8), ("Yes", "Yes", 1.0), ("yes.", "Yes", 1.0),
             ("No", "Yes", 0.0), ("Unanswerable", "Unanswerable", 1.0), ("BERT and ELMo", "BERT, ELMo", 0.8),
             ("an LSTM-based encoder", "LSTM based encoder", 0.4), ("", "Unanswerable", 0.0),
             ("They use 5 datasets: SST-2, MR, CR", "SST-2, MR, CR, SUBJ, MPQA", 0.5), ("the the the", "the", 0.0)]

    def test_token_f1_matches_official(self):
        for p, g, want in self.CASES:
            self.assertAlmostEqual(answer.token_f1_score(p, g), want, msg=(p, g))

    def test_normalize(self):
        self.assertEqual(answer.normalize_answer("an LSTM-based encoder"), "lstmbased encoder")

    def test_evaluate_aggregate_matches_official(self):
        gold = {"q1": [{"answer": "Unanswerable", "type": "none"}, {"answer": "BERT, ELMo", "type": "extractive"}],
                "q2": [{"answer": "Yes", "type": "boolean"}, {"answer": "No", "type": "boolean"}],
                "q3": [{"answer": "a recurrent network", "type": "abstractive"}]}
        pred = {"q1": "ELMo", "q2": "no", "q3": "recurrent neural network"}
        scored = {q: answer.qasper_answer_f1(pred[q], refs) for q, refs in gold.items()}
        self.assertAlmostEqual(sum(f for f, _ in scored.values()) / 3, 0.8222222222222223)
        self.assertEqual({q: t for q, (_, t) in scored.items()},
                         {"q1": "extractive", "q2": "boolean", "q3": "abstractive"})

    def test_references_like_official(self):
        anns = [{"answer": {"unanswerable": False, "extractive_spans": [], "free_form_answer": "", "yes_no": False}},
                {"answer": {"unanswerable": False, "extractive_spans": ["a", "b"], "free_form_answer": "x",
                            "yes_no": None}},
                {"answer": {"unanswerable": True, "extractive_spans": [], "free_form_answer": "", "yes_no": None}}]
        want = [("No", "boolean"), ("a, b", "extractive"), ("Unanswerable", "none")]
        self.assertEqual([(r["answer"], r["type"]) for r in answer.qasper_references(anns)], want)
        parquet = {"answer": [a["answer"] for a in anns], "annotation_id": ["1", "2", "3"]}
        self.assertEqual([(r["answer"], r["type"]) for r in answer.qasper_references(parquet)], want)

    def test_clean_answer(self):
        self.assertEqual(answer.clean_answer("Answer: Unanswerable.", "qasper"), "Unanswerable")
        self.assertEqual(answer.clean_answer("The question is unanswerable", "qasper"), "Unanswerable")
        self.assertEqual(answer.clean_answer("Yes\nbecause...", "qasper"), "Yes")


class TestContext(unittest.TestCase):
    CHUNKS = [{"text": "alpha beta gamma " * 40}, {"text": "delta epsilon " * 60}, {"text": "zeta eta theta " * 50}]

    def test_exact_budget(self):
        for b in (17, 100, 256, 300):
            _, n = answer.build_context([2, 0, 1], self.CHUNKS, b)
            self.assertEqual(n, b)
        total = sum(len(answer.ENC.encode(c["text"])) for c in self.CHUNKS)
        self.assertEqual(answer.build_context([0, 1, 2], self.CHUNKS, 10_000)[1], total)

    def test_rank_order_and_truncation(self):
        pieces = answer.select_text([1, 0], self.CHUNKS, 130)
        self.assertTrue(pieces[0].startswith("delta"))
        self.assertTrue(pieces[-1].startswith("alpha"))

    def test_same_selection_as_evaluate(self):
        try:
            import evaluate
        except Exception as e:  # pragma: no cover
            self.skipTest(f"evaluate.py not importable: {e}")
        for b in (64, 256, 512):
            order = [2, 1, 0]
            got = [evaluate.words(p) for p in answer.select_text(order, self.CHUNKS, b)]
            self.assertEqual(got, evaluate.retrieved_units(order, self.CHUNKS, budget=b))

    def test_generate_answer_uses_deterministic_settings(self):
        seen = {}

        def fake(spec, prompt, **kw):
            seen.update(kw, prompt=prompt)
            return {"text": "Answer: No", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0, "provider": "x"}

        r = answer.generate_answer("qasper", "Is it?", [0], self.CHUNKS, 32, "ollama:qwen2.5:7b", complete_fn=fake)
        self.assertEqual(r["answer"], "No")
        self.assertEqual((seen["temperature"], seen["seed"], seen["max_tokens"]), (0.0, 0, 64))
        self.assertEqual(r["context_tokens"], 32)
        self.assertIn("Unanswerable", seen["prompt"])


# --------------------------------------------------------------------------- judge

@unittest.skipUnless(HAS_JUDGE, "judge.py not present")
class TestJudge(unittest.TestCase):
    def setUp(self):
        import judge
        self.j = judge

    def test_parse_valid_forms(self):
        p = self.j.parse_nugget_labels
        self.assertEqual(p('{"labels":[{"id":2,"label":"no_support"},{"id":1,"label":"support"}]}', 2),
                         ["support", "no_support"])
        self.assertEqual(p('```json\n{"labels":["support","partial_support"]}\n```', 2),
                         ["support", "partial_support"])
        self.assertEqual(p('{"labels":[{"id":"1","label":"not_support"}]}', 1), ["no_support"])

    def test_parse_rejects(self):
        p, E = self.j.parse_nugget_labels, self.j.JudgeParseError
        for bad, n in [('{"labels":[{"id":1,"label":"support"}]}', 2),
                       ('{"labels":[{"id":1,"label":"support"},{"id":1,"label":"support"}]}', 2),
                       ('{"labels":[{"id":3,"label":"support"},{"id":1,"label":"support"}]}', 2),
                       ('{"labels":[{"id":1,"label":"maybe"}]}', 1), ("support", 1), ("{}", 1)]:
            with self.assertRaises(E, msg=bad):
                p(bad, n)

    def test_scores(self):
        s = self.j.nugget_scores(["support", "partial_support", "no_support", "support"])
        self.assertEqual((s["strict"], s["partial"]), (0.5, 0.625))

    def test_family_guard(self):
        with self.assertRaises(ValueError):
            self.j.Judge("openrouter:qwen/qwen3-235b-a22b-2507@deepinfra/fp8", other_models=["ollama:qwen2.5:7b"],
                         complete_fn=lambda *a, **k: None)

    def test_retry_on_bad_json_then_raise(self):
        replies = iter(["garbage", '{"labels":[{"id":1,"label":"support"}]}'])
        j = self.j.Judge("openrouter:meta-llama/llama-3.3-70b-instruct@deepinfra/turbo",
                         other_models=["ollama:qwen2.5:7b"], complete_fn=lambda *a, **k: {"text": next(replies)})
        r = j.judge_nuggets("q", "a", ["n1"])
        self.assertEqual((r["strict"], r["parse_attempts"], j.parse_failures), (1.0, 2, 1))
        j2 = self.j.Judge("gemini:gemini-3.1-flash-lite", complete_fn=lambda *a, **k: {"text": "nope"})
        with self.assertRaises(self.j.JudgeParseError):
            j2.judge_nuggets("q", "a", ["n1"])

    def test_controls_with_oracle_judge(self):
        def fake(spec, prompt, **kw):  # supports a nugget iff the answer contains its text
            ans = prompt.split("Answer:\n", 1)[1].split("\n\nNuggets:", 1)[0]
            nugs = prompt.split("Nuggets:\n", 1)[1].split("\n\nReturn", 1)[0].splitlines()
            labs = [{"id": i, "label": "support" if n.split(". ", 1)[1] in ans else "no_support"}
                    for i, n in enumerate(nugs, 1)]
            return {"text": json.dumps({"labels": labs})}

        j = self.j.Judge("openrouter:meta-llama/llama-3.3-70b-instruct@deepinfra/turbo", complete_fn=fake)
        items = [{"qid": str(i), "question": f"q{i}", "nuggets": [f"fact{i}a", f"fact{i}b"],
                  "gold_answer": f"fact{i}a and fact{i}b"} for i in range(5)]
        c = self.j.freshstack_controls(j, items)
        self.assertEqual((c["positive_strict_recall"], c["negative_strict_recall"]), (1.0, 0.0))
        self.assertTrue(c["positive_pass"] and c["negative_pass"])
        self.assertTrue(all(a != b for a, b in c["negative_pairs"]))

    def test_length_bias(self):
        rows = [{"system": "A", "score": s, "answer_words": w} for s, w in [(0, 5), (0.5, 10), (1, 20)]]
        out = self.j.length_bias(rows)
        self.assertAlmostEqual(out["A"]["spearman_score_length"], 1.0)


# --------------------------------------------------------------------------- stats

def synth(n_clusters, mean, sd, per=3, seed=0, cluster_sd=0.0):
    rng = np.random.default_rng(seed)
    return {f"c{i}": list(rng.normal(mean + rng.normal(0, cluster_sd), sd, per)) for i in range(n_clusters)}


class TestStats(unittest.TestCase):
    def test_bootstrap_mean_and_ci_order(self):
        d = synth(50, 1.0, 2.0)
        b = stats.cluster_bootstrap(d, n=4000)
        allv = [v for vs in d.values() for v in vs]
        self.assertAlmostEqual(b["mean"], float(np.mean(allv)))
        for m in ("percentile", "bca"):
            lo90, hi90 = b[m][0.90]
            lo95, hi95 = b[m][0.95]
            self.assertTrue(lo95 < lo90 < b["mean"] < hi90 < hi95)

    def test_bootstrap_matches_evaluate_style(self):
        # Percentile interval agrees with a naive loop implementation within MC error
        d = synth(30, 0.5, 1.0, seed=3)
        b = stats.cluster_bootstrap(d, n=10_000)
        keys = list(d)
        rng = np.random.default_rng(1)
        means = []
        for _ in range(4000):
            ks = rng.choice(len(keys), len(keys))
            vals = [v for k in ks for v in d[keys[k]]]
            means.append(np.mean(vals))
        self.assertAlmostEqual(b["percentile"][0.95][0], np.quantile(means, 0.025), delta=0.05)
        self.assertAlmostEqual(b["percentile"][0.95][1], np.quantile(means, 0.975), delta=0.05)

    def test_bootstrap_coverage_sanity(self):
        # 95% CIs over 200 synthetic datasets with true mean 0.3 should cover ~95%
        cover = {"percentile": 0, "bca": 0}
        reps = 200
        for s in range(reps):
            d = synth(40, 0.3, 1.0, per=2, seed=100 + s, cluster_sd=0.3)
            b = stats.cluster_bootstrap(d, n=1000, seed=s, levels=(0.95,))
            for m in cover:
                lo, hi = b[m][0.95]
                cover[m] += lo <= 0.3 <= hi
        for m, c in cover.items():
            self.assertGreater(c / reps, 0.88, m)
            self.assertLess(c / reps, 0.995, m)

    def test_bca_close_to_percentile_for_symmetric(self):
        b = stats.cluster_bootstrap(synth(200, 0, 1, seed=5), n=10_000)
        for i in (0, 1):
            self.assertAlmostEqual(b["bca"][0.95][i], b["percentile"][0.95][i], delta=0.02)

    def test_bca_shifts_for_skewed(self):
        rng = np.random.default_rng(9)
        d = {i: [float(x)] for i, x in enumerate(rng.exponential(1.0, 60))}
        b = stats.cluster_bootstrap(d, n=10_000)
        self.assertGreater(b["a"], 0)  # right skew -> positive acceleration
        self.assertGreater(b["bca"][0.95][1], b["percentile"][0.95][1])

    def test_stratified(self):
        d = {f"a{i}": [1.0] for i in range(10)} | {f"b{i}": [3.0] for i in range(30)}
        strata = {k: k[0] for k in d}
        b = stats.cluster_bootstrap(d, n=2000, strata=strata)
        self.assertAlmostEqual(np.ptp(b["boot"]), 0.0)  # strata sizes fixed -> mean fixed

    def test_tost_verdicts(self):
        eq = stats.tost(synth(300, 0.0, 2.0, seed=1), margin=3, n=4000)
        self.assertEqual(eq["verdict"], "equivalent")
        self.assertLess(eq["p_tost"], 0.05)
        self.assertLess(eq["p_tost_t"], 0.05)
        diff = stats.tost(synth(300, 8.0, 2.0, seed=2), margin=3, n=4000)
        self.assertEqual(diff["verdict"], "different")
        self.assertGreater(diff["p_tost"], 0.5)
        inc = stats.tost(synth(8, 1.0, 6.0, seed=3), margin=3, n=4000)
        self.assertEqual(inc["verdict"], "inconclusive")
        both = stats.tost(synth(400, 1.0, 1.0, seed=4), margin=3, n=4000)
        self.assertEqual((both["verdict"], both["label"]), ("equivalent", "different_but_equivalent"))

    def test_tost_p_consistent_with_ci(self):
        for seed in range(12):
            d = synth(40, float(seed % 4), 4.0, seed=seed)
            for m in ("bca", "percentile"):
                r = stats.tost(d, margin=3, n=4000, seed=seed, method=m)
                if abs(r["p_tost"] - 0.05) > 0.005:  # away from the MC-noise boundary
                    self.assertEqual(r["p_tost"] < 0.05, r["equivalent"], (seed, m, r["p_tost"], r["ci90"]))

    def test_t_cdf(self):
        self.assertAlmostEqual(stats.t_cdf(2.228138852, 10), 0.975, places=6)
        self.assertAlmostEqual(stats.t_cdf(-1.812461123, 10), 0.05, places=6)
        self.assertAlmostEqual(stats.t_cdf(0.0, 5), 0.5, places=12)
        self.assertAlmostEqual(stats.t_cdf(1.959963985, 1e7), 0.975, places=5)

    def test_sign_flip_exact(self):
        # 4 clusters, all positive: only the all-plus and all-minus flips are as extreme -> p = 2/16
        d = {"a": [1.0], "b": [2.0], "c": [3.0], "e": [4.0]}
        r = stats.sign_flip_test(d, n=10_000)
        self.assertTrue(r["exact"])
        self.assertAlmostEqual(r["p"], 2 / 16)

    def test_sign_flip_mc(self):
        null = stats.sign_flip_test(synth(60, 0.0, 1.0, seed=8), n=5000)
        alt = stats.sign_flip_test(synth(60, 1.0, 1.0, seed=8), n=5000)
        self.assertGreater(null["p"], 0.05)
        self.assertLess(alt["p"], 0.001)
        self.assertGreaterEqual(alt["p"], 1 / 5001)

    def test_sign_flip_null_calibration(self):
        rej = sum(stats.sign_flip_test(synth(30, 0.0, 1.0, seed=500 + s), n=999, seed=s)["p"] <= 0.05
                  for s in range(200))
        self.assertLess(rej / 200, 0.10)

    def test_holm(self):
        r = stats.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(r["p_adj"], [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(r["reject"], [True, False, False, True])
        self.assertEqual(stats.holm([0.9, 0.8])["p_adj"], [1.0, 1.0])

    def test_kappa(self):
        # Classic 2x2: 20/5/10/15 -> po=0.7, pe=0.5 -> kappa 0.4
        a = ["y"] * 25 + ["n"] * 25
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        self.assertAlmostEqual(stats.cohen_kappa(a, b), 0.4)
        self.assertAlmostEqual(stats.cohen_kappa(a, a), 1.0)
        lab = [0, 1, 2]
        x, y = [0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 0]
        # hand-computed: po_w(linear)=1-(2*0.5)/6, pe from marginals
        self.assertGreater(stats.cohen_kappa(x, y, lab, "linear"), stats.cohen_kappa(x, y, lab))
        self.assertAlmostEqual(stats.cohen_kappa(x, y, lab), (4 / 6 - 1 / 3) / (1 - 1 / 3))

    def test_spearman(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(stats.spearman([1, 2, 2, 3], [1, 2, 3, 4]), 0.9486832980505138)


if __name__ == "__main__":
    unittest.main()
