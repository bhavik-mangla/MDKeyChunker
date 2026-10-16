"""Offline tests for Amendment 1 (e2e_freshstack.py, build_scaling.py). LLMs, judge and
retrievers are mocked; nothing touches the network, Ollama or the real caches.

Run:  .venv/bin/python -m unittest tests/test_amendment1.py -v
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_scaling  # noqa: E402
import build_v2  # noqa: E402
import e2e_freshstack as fs  # noqa: E402
import judge as judge_mod  # noqa: E402
import llm_api  # noqa: E402

MD = """# Title

## Intro

Alpha beta gamma delta epsilon. """ + "Words about retrieval. " * 30 + """

## Method

We use BM25 and dense embeddings. """ + "More method text here. " * 30


def fake_rank(retriever, docs, queries):
    return [list(range(len(docs))) for _ in queries]


class FakeGen:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, spec, prompt, json_mode=False, max_tokens=0, temperature=0, seed=0, reasoning=None,
                 tag="", **kw):
        with self.lock:
            self.calls.append({"spec": spec, "max_tokens": max_tokens, "tag": tag, "prompt": prompt})
        return {"text": f"answer {len(self.calls)}", "in_tokens": 10, "out_tokens": 3, "cost_usd": 0.0,
                "provider": "fake", "model": "fake"}


class OracleJudgeComplete:
    """Labels a nugget 'support' iff its text appears in the answer."""

    def __init__(self, bad_first: int = 0):
        self.calls = []
        self.bad_first = bad_first
        self.lock = threading.Lock()

    def __call__(self, spec, prompt, json_mode=False, max_tokens=0, temperature=0, seed=0, tag="", **kw):
        with self.lock:
            self.calls.append({"spec": spec, "json": json_mode, "temperature": temperature, "seed": seed})
            n = len(self.calls)
        if n <= self.bad_first:
            return {"text": "not json"}
        ans = prompt.split("Answer:\n", 1)[1].split("\n\nNuggets:\n", 1)[0]
        nug = prompt.split("Nuggets:\n", 1)[1].split("\n\nReturn only JSON", 1)[0].splitlines()
        labels = []
        for line in nug:
            i, text = line.split(". ", 1)
            labels.append({"id": int(i), "label": "support" if text in ans else "no_support"})
        return {"text": json.dumps({"labels": labels})}


def make_cache(root: Path, systems: dict, doc_ids: list[str]) -> None:
    for label in systems:
        tag, variant, _ = build_v2_systems()[label]
        for d in doc_ids:
            p = build_v2.cache_path(tag, variant, d, root)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"chunks": [{"text": f"{d} chunk {k} " + "word " * 50, "section": "s",
                                                 "title": "t", "summary": "", "keywords": [], "questions": [],
                                                 "context": "c"} for k in range(3)]}))


def build_v2_systems():
    import evaluate_v2
    return evaluate_v2.SYSTEMS_V2


QUESTIONS = {
    "laravel": [{"qid": "q1", "question": "How?", "nuggets": ["use foo", "call bar"], "answer": "use foo and call bar",
                 "domain": "laravel"},
                {"qid": "q2", "question": "Why?", "nuggets": ["cache it"], "answer": "cache it", "domain": "laravel"}],
    "angular": [{"qid": "q3", "question": "What?", "nuggets": ["signals"], "answer": "signals", "domain": "angular"}],
}


# --------------------------------------------------------------------------- e2e FreshStack

class TestAnswerCaching(unittest.TestCase):
    def test_answers_cached_and_resumed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_cache(td / "c", ["S", "E"], ["d1", "d2"])
            plan = {"groups": {"laravel": ["d1", "d2"]}, "systems": ["S", "E"], "strong_groups": {}}
            gen = FakeGen()
            dqs = {"laravel": QUESTIONS["laravel"]}
            r1 = fs.generate(dqs, plan, td / "c", ans_cache=td / "a", complete_fn=gen, rank_fn=fake_rank,
                             log=lambda s: None)
            self.assertEqual(len(r1["rows"]), 4)
            self.assertEqual(len(gen.calls), 4)
            self.assertTrue(all(c["spec"] == fs.GEN_SPEC for c in gen.calls))
            self.assertTrue(all(c["max_tokens"] == 512 for c in gen.calls))  # GEN_SETTINGS freshstack
            self.assertTrue(fs.answer_file("S", "q1", td / "a").exists())
            self.assertTrue(fs.answer_file("E", "q2", td / "a").exists())
            # resume: nothing regenerated
            r2 = fs.generate(dqs, plan, td / "c", ans_cache=td / "a", complete_fn=gen, rank_fn=fake_rank,
                             log=lambda s: None)
            self.assertEqual(len(gen.calls), 4)
            self.assertEqual(sorted(r["answer"] for r in r1["rows"]), sorted(r["answer"] for r in r2["rows"]))
            # partial resume: remove one answer, only it is regenerated
            fs.answer_file("E", "q1", td / "a").unlink()
            fs.generate(dqs, plan, td / "c", ans_cache=td / "a", complete_fn=gen, rank_fn=fake_rank,
                        log=lambda s: None)
            self.assertEqual(len(gen.calls), 5)

    def test_context_is_1024_tokens_of_source_text(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_cache(td / "c", ["S"], [f"d{i}" for i in range(20)])
            gen = FakeGen()
            r = fs.answer_domain("laravel", QUESTIONS["laravel"][:1], "S", [f"d{i}" for i in range(20)],
                                 td / "c", ans_cache=td / "a", complete_fn=gen, rank_fn=fake_rank)
            self.assertEqual(r[0]["context_tokens"], 1024)

    def test_strong_only_on_covered_domains(self):
        plan = {"groups": {"laravel": ["d1"], "angular": ["d2"]}, "systems": ["S", "E", "E-strong"],
                "strong_groups": {"E-strong": ["laravel"]}}
        jobs = fs.jobs_for(plan)
        self.assertIn(("laravel", "E-strong"), jobs)
        self.assertNotIn(("angular", "E-strong"), jobs)

    def test_nuggets_string_parsed(self):
        self.assertEqual(fs._as_list("['a', 'b']"), ["a", "b"])
        self.assertEqual(fs._as_list('["a"]'), ["a"])


class TestJudgeWiring(unittest.TestCase):
    def test_spec_pinned_and_family_guard(self):
        s = llm_api.ModelSpec.parse(fs.JUDGE_SPEC)
        self.assertEqual(s.backend, "openrouter")
        self.assertEqual(s.providers, ("deepinfra/turbo",))
        self.assertEqual(llm_api.model_family(fs.JUDGE_SPEC), "llama")
        fams = {llm_api.model_family(m) for m in fs.OTHER_MODELS}
        self.assertIn("qwen", fams)
        self.assertIn("gpt-oss", fams)
        fs.CachedJudge(fs.JUDGE_SPEC, other_models=fs.OTHER_MODELS, complete_fn=OracleJudgeComplete())
        with self.assertRaises(ValueError):
            fs.CachedJudge("dashscope:qwen3-235b-a22b-instruct-2507", other_models=fs.OTHER_MODELS)

    def test_cached_judge_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            fake = OracleJudgeComplete()
            jg = fs.CachedJudge(fs.JUDGE_SPEC, other_models=fs.OTHER_MODELS, complete_fn=fake, cache=Path(td))
            qmap = {q["qid"]: q for qs in QUESTIONS.values() for q in qs}
            rows = [{"qid": "q1", "system": "E", "answer": "use foo"},
                    {"qid": "q1", "system": "S", "answer": "use foo and call bar"}]
            out = fs.judge_rows(jg, rows, qmap, workers=2, log=lambda s: None)
            by = {r["system"]: r for r in out}
            self.assertEqual(by["E"]["labels"], ["support", "no_support"])
            self.assertAlmostEqual(by["E"]["strict"], 0.5)
            self.assertAlmostEqual(by["S"]["strict"], 1.0)
            self.assertTrue(all(c["temperature"] == 0.0 and c["json"] for c in fake.calls))
            self.assertTrue(all(c["spec"] == fs.JUDGE_SPEC for c in fake.calls))
            n = len(fake.calls)
            fs.judge_rows(jg, rows, qmap, log=lambda s: None)
            self.assertEqual(len(fake.calls), n)  # all cache hits
            self.assertEqual(jg.hits, 2)

    def test_parse_failure_recorded_not_cached(self):
        with tempfile.TemporaryDirectory() as td:
            fake = OracleJudgeComplete(bad_first=judge_mod.MAX_PARSE_ATTEMPTS)
            jg = fs.CachedJudge(fs.JUDGE_SPEC, complete_fn=fake, cache=Path(td))
            qmap = {"q2": QUESTIONS["laravel"][1]}
            out = fs.judge_rows(jg, [{"qid": "q2", "system": "S", "answer": "cache it"}], qmap, log=lambda s: None)
            # counted as no_support (plan v2 sec. 7), flagged, and not cached
            self.assertEqual((out[0]["strict"], out[0]["partial"], out[0]["judge_failed"]), (0.0, 0.0, True))
            self.assertEqual(list(Path(td).iterdir()), [])
            out = fs.judge_rows(jg, [{"qid": "q2", "system": "S", "answer": "cache it"}], qmap, log=lambda s: None)
            self.assertEqual(out[0]["strict"], 1.0)

    def test_estimate(self):
        e = fs.estimate_judge([("q", "a " * 100, ["n1", "n2"])])
        self.assertEqual(e["calls"], 1)
        self.assertGreater(e["in_tokens"], 100)
        self.assertLess(e["usd"], 0.001)


class TestControls(unittest.TestCase):
    def test_controls_with_oracle(self):
        with tempfile.TemporaryDirectory() as td:
            items = fs.control_items(QUESTIONS)
            self.assertEqual(sorted(i["qid"] for i in items), ["q1", "q2", "q3"])
            self.assertEqual(items, fs.control_items(QUESTIONS))  # deterministic order
            fake = OracleJudgeComplete()
            jg = fs.CachedJudge(fs.JUDGE_SPEC, complete_fn=fake, cache=Path(td))
            c = fs.run_controls(jg, items, workers=3)
            self.assertEqual(c["positive_strict_recall"], 1.0)
            self.assertEqual(c["negative_strict_recall"], 0.0)
            self.assertTrue(c["positive_pass"] and c["negative_pass"])
            self.assertEqual(len(fake.calls), 6)  # pre-warm only; freshstack_controls hits the cache
            self.assertEqual(fs.acceptance(c, None)["status"], "exploratory_no_spotcheck")
            self.assertEqual(fs.acceptance(c, {"n": 10, "kappa": 0.7})["status"], "secondary")
            self.assertEqual(fs.acceptance(c, {"n": 10, "kappa": 0.4})["status"], "failed_kappa")
            self.assertEqual(fs.acceptance({**c, "negative_pass": False}, None)["status"], "failed_controls")

    def test_questions_without_gold_excluded(self):
        q = {"x": [{"qid": "a", "question": "?", "nuggets": ["n"], "answer": ""},
                   {"qid": "b", "question": "?", "nuggets": ["n"], "answer": "n"}]}
        self.assertEqual([i["qid"] for i in fs.control_items(q)], ["b"])


class TestAnalysis(unittest.TestCase):
    def test_tost_and_length_bias(self):
        judged = []
        strata = {}
        for k in range(40):
            qid = f"q{k}"
            strata[qid] = "laravel" if k % 2 else "angular"
            for s, v in (("E", 0.5 + (k % 3) * 0.01), ("S+T", 0.5), ("CR", 0.2)):
                judged.append({"qid": qid, "system": s, "strict": v, "partial": v, "answer_words": 10 + k})
        res = fs.analyze(judged, strata, n=500)
        rows = {(r["a"], r["b"], r["metric"]): r for r in res["comparisons"]}
        self.assertEqual(rows[("E", "S+T", "nugget_strict")]["verdict"], "equivalent")
        self.assertEqual(rows[("E", "S+T", "nugget_strict")]["margin"], 0.05)
        self.assertEqual(rows[("E", "S+T", "nugget_strict")]["family"], "secondary")
        self.assertEqual(rows[("E", "CR", "nugget_strict")]["verdict"], "different")
        self.assertNotIn("margin", rows[("E", "S+T", "nugget_partial")])
        self.assertAlmostEqual(res["means"]["CR"]["strict"]["mean"], 0.2)


class TestSpotcheck(unittest.TestCase):
    def _judged(self):
        qmap = {}
        judged = []
        labs = list(judge_mod.LABELS)
        for k in range(60):
            qid = f"q{k}"
            qmap[qid] = {"qid": qid, "question": f"question {k}", "nuggets": [f"n{k}a", f"n{k}b"]}
            for s in ("E", "S"):
                judged.append({"qid": qid, "system": s, "answer": f"ans {s} {k}",
                               "labels": [labs[k % 3], labs[(k + 1) % 3]]})
        return judged, qmap

    def test_sheet_blind_key_roundtrip_kappa(self):
        judged, qmap = self._judged()
        with tempfile.TemporaryDirectory() as td:
            sheet, key = Path(td) / "sheet.csv", Path(td) / "key.csv"
            self.assertEqual(fs.make_spotcheck(judged, qmap, sheet, key), 120)
            rows = list(csv.DictReader(sheet.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(list(rows[0].keys()), fs.SHEET_COLS)
            self.assertEqual(len({r["item_id"] for r in rows}), 120)
            self.assertTrue(all(r["author_label"] == "" for r in rows))
            text = sheet.read_text()
            for hidden in ("support", "system"):
                self.assertNotIn(hidden, text.split("\n", 1)[1])  # no labels or systems in the sheet body
            # deterministic, and never clobbers a sheet in progress
            self.assertEqual(fs.make_spotcheck(judged, qmap, sheet, key), -1)
            # unfilled -> n=0
            self.assertEqual(fs.score_spotcheck(sheet, key)["n"], 0)
            # fill with the judge labels -> kappa 1; with short codes
            kmap = {r["item_id"]: r["judge_label"] for r in csv.DictReader(key.read_text(encoding="utf-8").splitlines())}
            short = {"support": "s", "partial_support": "p", "no_support": "n"}
            for r in rows:
                r["author_label"] = short[kmap[r["item_id"]]]
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fs.SHEET_COLS)
                w.writeheader()
                w.writerows(rows)
            k = fs.score_spotcheck(sheet, key)
            self.assertEqual(k["n"], 120)
            self.assertTrue(k["complete"])
            self.assertAlmostEqual(k["kappa"], 1.0)
            self.assertAlmostEqual(k["agreement"], 1.0)
            # flip half the labels -> kappa drops below threshold
            for i, r in enumerate(rows):
                if i % 2:
                    r["author_label"] = "no_support" if kmap[r["item_id"]] != "no_support" else "support"
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fs.SHEET_COLS)
                w.writeheader()
                w.writerows(rows)
            self.assertLess(fs.score_spotcheck(sheet, key)["kappa"], fs.KAPPA_MIN)

    def test_sheet_deterministic(self):
        judged, qmap = self._judged()
        with tempfile.TemporaryDirectory() as td:
            a = [Path(td) / f"{x}.csv" for x in ("s1", "k1", "s2", "k2")]
            fs.make_spotcheck(judged, qmap, a[0], a[1])
            fs.make_spotcheck(list(reversed(judged)), qmap, a[2], a[3])
            self.assertEqual(a[0].read_text(), a[2].read_text())
            self.assertEqual(a[1].read_text(), a[3].read_text())


# --------------------------------------------------------------------------- scaling

class TestScaling(unittest.TestCase):
    def test_sample_deterministic(self):
        ids = [f"{1700 + i}.{i:05d}" for i in range(300)]
        s1 = build_scaling.sample_ids(list(reversed(ids)))
        s2 = build_scaling.sample_ids(ids)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 30)
        self.assertEqual(s1, __import__("random").Random(2027).sample(sorted(ids), 30))

    def test_real_selection_sample_stable(self):
        sel = json.loads((ROOT / "data" / "v2" / "selection.json").read_text())
        ids = build_scaling.sample_ids(sel["qasper"]["paper_ids"])
        self.assertEqual(len(set(ids)), 30)
        self.assertTrue(set(ids) <= set(sel["qasper"]["paper_ids"]))

    def test_separate_frozen_config(self):
        self.assertNotEqual(build_scaling.FROZEN_SCALING.resolve(), build_v2.FROZEN.resolve())
        before = build_v2.FROZEN.read_bytes() if build_v2.FROZEN.exists() else None
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "frozen_scaling.json"
            cfg = build_scaling.scaling_config(["a", "b"])
            self.assertEqual(cfg["model"]["spec"], "dashscope:qwen3-235b-a22b-instruct-2507")
            self.assertEqual(cfg["model"]["temperature"], 0.0)
            self.assertEqual(cfg["model"]["seed"], 0)
            self.assertFalse(cfg["model"]["thinking"])
            build_v2.check_frozen(cfg, p, ignore_git=True)
            self.assertTrue(p.exists())
            build_v2.check_frozen(build_scaling.scaling_config(["a", "b"]), p, ignore_git=True)
            with self.assertRaises(build_v2.FrozenConfigMismatch):
                build_v2.check_frozen(build_scaling.scaling_config(["a", "c"]), p, ignore_git=True)
        after = build_v2.FROZEN.read_bytes() if build_v2.FROZEN.exists() else None
        self.assertEqual(before, after)
        with self.assertRaises(SystemExit):
            build_scaling.main(["--frozen", str(build_v2.FROZEN), "--dry-run"])

    def test_dashscope_thinking_off(self):
        # llm_api maps reasoning="off" to enable_thinking False
        seen = {}

        class Http:
            def post(self, url, json=None, headers=None, timeout=None):
                seen.update(json)

                class R:
                    status_code = 200
                    headers = {}
                    text = ""

                    @staticmethod
                    def json():
                        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                return R()

        with tempfile.TemporaryDirectory() as td:
            c = llm_api.LLMClient(ledger=llm_api.Ledger(Path(td) / "l.jsonl"), http=Http(),
                                  secret=lambda n: "k")
            c.complete(build_scaling.MODEL.spec, "p", json_mode=True, max_tokens=10, temperature=0.0, seed=0,
                       reasoning=build_scaling.MODEL.reasoning)
        self.assertIs(seen["enable_thinking"], False)
        self.assertEqual(seen["temperature"], 0.0)
        self.assertEqual(seen["seed"], 0)

    def test_builder_writes_only_qwen235b_enr_and_retries_parse(self):
        from tests.test_pipeline_v2 import FakeComplete
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            fake = FakeComplete(bad_json=2)
            b = build_scaling.ScalingBuilder("qasper", build_scaling.TAG, ["enr"], cache=cache, complete_fn=fake,
                                             log=lambda s: None)
            res = b.run([{"id": "p1", "markdown": MD}], workers=1)
            self.assertEqual(res["done"], 1)
            files = sorted(str(p.relative_to(cache)) for p in cache.rglob("*.json"))
            self.assertEqual(files, ["qwen235b/enr/p1.json"])
            st = json.loads((cache / "qwen235b/enr/p1.json").read_text())["stats"]
            self.assertEqual(st["parse_retries"], 2)
            self.assertEqual(st["spec"], build_scaling.MODEL.spec)
            self.assertTrue(all(c["spec"] == build_scaling.MODEL.spec for c in fake.calls))
            self.assertEqual([c["seed"] for c in fake.calls[:3]], [0, 1, 2])
            n = len(fake.calls)
            b.run([{"id": "p1", "markdown": MD}], workers=1)  # cached -> no calls
            self.assertEqual(len(fake.calls), n)

    def test_token_guard_stops_cleanly(self):
        from tests.test_pipeline_v2 import FakeComplete
        used = {"n": 0}
        inner = FakeComplete()

        def counting(spec, prompt, **kw):
            r = inner(spec, prompt, **kw)
            used["n"] += 1500
            return r

        with tempfile.TemporaryDirectory() as td:
            guard = build_scaling.TokenGuard(cap=4000, complete_fn=counting, used_fn=lambda: used["n"])
            b = build_scaling.ScalingBuilder("qasper", build_scaling.TAG, ["enr"], cache=Path(td),
                                             complete_fn=guard, log=lambda s: None)
            res = b.run([{"id": f"p{i}", "markdown": MD} for i in range(3)], workers=1)
            self.assertTrue(res["budget_stop"])
            self.assertLessEqual(used["n"], 4000 + 1500)
            self.assertEqual(guard.inflight, 0)
            done = sorted(x.stem for x in Path(td).rglob("*.json"))
            self.assertLess(len(done), 3)  # the paper that hit the cap is not cached
            self.assertEqual(done, [f"p{i}" for i in range(len(done))])

    def test_token_guard_reserves_inflight(self):
        g = build_scaling.TokenGuard(cap=1000, complete_fn=lambda *a, **k: {}, used_fn=lambda: 0)
        g.inflight = 900
        with self.assertRaises(llm_api.BudgetExceeded):
            g("dashscope:x", "hello", max_tokens=200)

    def test_evaluate_requires_build_complete(self):
        with mock.patch.object(build_scaling, "build_complete", return_value=False):
            self.assertEqual(build_scaling.main(["--evaluate"]), 2)

    def test_evaluate_scaling_offline(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            for label, (tag, variant, _) in build_scaling.EVAL_SYSTEMS.items():
                p = build_v2.cache_path(tag, variant, "p1", cache)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"chunks": [{"text": "alpha beta gamma delta epsilon zeta", "section": "",
                                                     "title": "", "summary": "", "keywords": [], "questions": []},
                                                    {"text": "other words here entirely", "section": ""}]}))
            papers = [{"id": "p1", "questions": [{"qid": "x", "question": "alpha?",
                                                  "evidence": ["alpha beta gamma delta epsilon"]}]},
                      {"id": "p2", "questions": [{"qid": "y", "question": "?", "evidence": ["z"]}]}]
            res = build_scaling.evaluate_scaling(papers, cache, rank_fn=fake_rank, n=200)
            self.assertEqual(res["papers_used"], ["p1"])
            self.assertEqual(res["papers_skipped"], ["p2"])
            self.assertIn("hybrid|E-qwen235b|rec@512t", res["means"])
            self.assertIn("bm25|E-qwen235b|rec@512t", res["means"])


if __name__ == "__main__":
    unittest.main()
