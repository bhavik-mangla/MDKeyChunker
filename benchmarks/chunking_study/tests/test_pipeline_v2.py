"""Offline tests for the v2 pipeline (select/build/evaluate/e2e/orchestrate). No network:
every LLM call goes to a fake complete(); embeddings are replaced by a hash embedder.

Run:  .venv/bin/python -m unittest tests/test_pipeline_v2.py -v
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import build_v2  # noqa: E402
import e2e_v2  # noqa: E402
import evaluate  # noqa: E402
import evaluate_v2  # noqa: E402
import llm_api  # noqa: E402
import orchestrate_v2  # noqa: E402
import select_v2  # noqa: E402

logging.getLogger("mdkeychunker").setLevel(logging.ERROR)

MD = """# Title

Intro paragraph about retrieval and chunking methods for documents.

## Methods

We use BM25 and dense embeddings. The hybrid retriever fuses both rankings.

```python
def f(x):

    return x
```

## Results

The structural chunker improves recall on scientific papers by a margin.
"""


def fake_embed(model_key, texts, is_query):
    out = []
    for t in texts:
        v = np.frombuffer(hashlib.sha256(f"{model_key}|{t}".encode()).digest(), dtype=np.uint8).astype(np.float32)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.vstack(out)


def ok_enrich_json(prompt: str) -> dict:
    h = hashlib.sha1(prompt.encode()).hexdigest()[:6]
    return {"title": f"t{h}", "summary": "s", "keywords": ["k1", "k2"], "entities": [],
            "questions": ["q?"], "key": f"key {h}", "related_keys": []}


class FakeComplete:
    """Records calls; json mode returns a valid enrichment unless told to fail."""

    def __init__(self, bad_json: int = 0, budget_after: int | None = None, delay: float = 0.0):
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.bad_json = bad_json
        self.budget_after = budget_after
        self.delay = delay

    def __call__(self, spec, prompt, json_mode=False, max_tokens=0, temperature=0, seed=0, reasoning=None,
                 tag="", **kw):
        with self.lock:
            n = len(self.calls)
            self.calls.append({"spec": spec, "prompt": prompt, "json": json_mode, "seed": seed,
                               "reasoning": reasoning, "max_tokens": max_tokens, "tag": tag,
                               "thread": threading.get_ident()})
        if self.budget_after is not None and n >= self.budget_after:
            raise llm_api.BudgetExceeded("cap")
        if self.delay:
            time.sleep(self.delay)
        if json_mode:
            parsed = None if n < self.bad_json else ok_enrich_json(prompt)
            return {"text": json.dumps(parsed), "parsed_json": parsed, "in_tokens": 10, "out_tokens": 5,
                    "cost_usd": 0.001, "provider": "fake"}
        chunk = prompt.split("<chunk>\n", 1)[-1].split("\n</chunk>", 1)[0]
        return {"text": f"ctx for {chunk[:20]}", "parsed_json": None, "in_tokens": 10, "out_tokens": 5,
                "cost_usd": 0.0, "provider": "fake"}


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# --------------------------------------------------------------------------- selection

class TestSelection(Tmp):
    def _write_fixture(self):
        v2 = self.tmp / "v2"
        v2.mkdir()
        papers = [{"id": f"p{i}", "markdown": MD, "questions": [{"qid": f"q{i}", "evidence": ["x"]}]}
                  for i in range(3)]
        (v2 / "qasper_v2.json").write_text(json.dumps(papers))
        docs = []
        for dom, n in (("laravel", 6), ("angular", 12), ("yolo", 9)):
            paths = [f"d/{dom}{j}.md" for j in range(n)]
            docs += [{"id": select_v2.fs_doc_id(dom, p), "domain": dom, "path": p, "markdown": MD} for p in paths]
            gold = paths[:5] if dom != "laravel" else paths[:2]
            qs = [{"qid": f"{dom}-q{j}", "evidence": [{"path": g}]} for j, g in enumerate(gold)]
            if dom == "laravel":
                qs.append({"qid": "pilot-1", "evidence": [{"path": paths[0]}]})
            (v2 / f"freshstack_{dom}.json").write_text(json.dumps({"questions": qs, "commit": "abc"}))
        (v2 / "freshstack_v2_docs.json").write_text(json.dumps(docs))
        pilot = self.tmp / "pilot.json"
        pilot.write_text(json.dumps({"questions": [{"qid": "pilot-1"}]}))
        return v2, pilot

    def test_distractor_sample_deterministic_and_sized(self):
        gold = [f"g{i}" for i in range(7)]
        allp = gold + [f"n{i}" for i in range(30)]
        a = select_v2.sample_distractors(gold, allp)
        self.assertEqual(a, select_v2.sample_distractors(gold, list(reversed(allp))))
        self.assertEqual(len(a), round(0.5 * 7))
        self.assertFalse(set(a) & set(gold))

    def test_select_rules_and_determinism(self):
        v2, pilot = self._write_fixture()
        s1 = select_v2.select(v2, pilot)
        s2 = select_v2.select(v2, pilot)
        self.assertEqual(json.dumps(s1, sort_keys=True), json.dumps(s2, sort_keys=True))
        self.assertEqual(s1["qasper"]["paper_ids"], ["p0", "p1", "p2"])
        fs = s1["freshstack"]
        self.assertNotIn("pilot-1", fs["laravel"]["qids"])
        self.assertEqual(len(fs["laravel"]["doc_ids"]), 6)                    # full corpus
        for dom in ("angular", "yolo"):
            self.assertEqual(len(fs[dom]["gold_doc_ids"]), 5)
            self.assertEqual(len(fs[dom]["distractor_doc_ids"]), round(0.5 * 5))
            self.assertTrue(set(fs[dom]["gold_doc_ids"]) <= set(fs[dom]["doc_ids"]))
        c = s1["counts"]
        self.assertEqual(c["freshstack"]["docs"], 6 + 7 + 7)
        self.assertEqual(c["qasper"]["struct_chunks"], 3 * s1["doc_stats"]["p0"]["struct_chunks"])

    def test_real_selection_counts_if_present(self):
        p = ROOT / "data" / "v2" / "selection.json"
        if not p.exists():
            self.skipTest("selection.json not built")
        c = json.loads(p.read_text())["counts"]
        self.assertEqual(c["qasper"]["docs"], 300)
        self.assertEqual(c["freshstack/laravel"]["docs"], 98)
        self.assertEqual(c["freshstack/angular"]["docs"], 177)
        self.assertEqual(c["freshstack/yolo"]["docs"], 102)


# --------------------------------------------------------------------------- chunkers

class TestChunkers(unittest.TestCase):
    def test_paragraphs_keep_fences(self):
        ps = build_v2.paragraphs(MD)
        fence = [p for p in ps if p.startswith("```python")]
        self.assertEqual(len(fence), 1)
        self.assertIn("return x", fence[0])            # the blank line inside the fence did not split it

    def test_parapack(self):
        md = "\n\n".join(f"# H{i}\n\n" + ("word " * 40).strip() for i in range(20))
        chunks = build_v2.parapack(md, max_size=500)
        paras = build_v2.paragraphs(md)
        self.assertEqual("\n\n".join(c["text"] for c in chunks), "\n\n".join(paras))  # lossless, in order
        for c in chunks:
            self.assertLessEqual(len(c["text"]), 500)
        # greedy: adding the next paragraph would have overflowed
        for a, b in zip(chunks, chunks[1:]):
            nxt = build_v2.paragraphs(b["text"])[0]
            self.assertGreater(len(a["text"] + "\n\n" + nxt), 500)
        # headers do not force a boundary: some chunk holds two headers
        self.assertTrue(any(c["text"].count("# H") >= 2 for c in chunks))

    def test_parapack_oversize_paragraph_whole(self):
        big = "x" * 3000
        chunks = build_v2.parapack(f"a\n\n{big}\n\nb", max_size=1500)
        self.assertEqual([c["text"] for c in chunks], ["a", big, "b"])

    def test_fixedlen(self):
        md = MD * 10
        toks = build_v2.ENC.encode(md)
        chunks = build_v2.fixed_len(md, 37)
        lens = [len(build_v2.ENC.encode(c["text"])) for c in chunks]
        self.assertEqual(len(chunks), -(-len(toks) // 37))
        self.assertEqual("".join(c["text"] for c in chunks), md)                 # no overlap, nothing lost
        self.assertTrue(all(abs(n - 37) <= 2 for n in lens[:-1]))

    def test_median_struct_tokens(self):
        from mdkeychunker.chunker import MarkdownChunker
        from mdkeychunker.config import Config
        lens = sorted(len(build_v2.ENC.encode(c.text)) for c in MarkdownChunker(Config()).chunk(MD * 3))
        m = build_v2.median_struct_tokens([MD * 3])
        self.assertTrue(lens[0] <= m <= lens[-1])


# --------------------------------------------------------------------------- builds

class TestBuild(Tmp):
    def builder(self, tag, variants, fake, **kw):
        return build_v2.Builder("qasper", tag, variants, cache=self.tmp, complete_fn=fake, fixedlen_size=50,
                                log=lambda s: None, **kw)

    def test_cache_separation_by_model_tag(self):
        doc = {"id": "d1", "markdown": MD}
        f_local, f_strong = FakeComplete(), FakeComplete()
        self.builder("local", ["fixed512", "fixedtok", "struct", "parapack", "fixedlen", "enr", "cr"],
                     f_local).build_doc(doc)
        self.builder("strong", ["enr", "cr"], f_strong).build_doc(doc)
        for v in ("fixed512", "fixedtok", "struct", "parapack", "fixedlen"):
            self.assertTrue((self.tmp / "base" / v / "d1.json").exists())
        for tag in ("local", "strong"):
            for v in ("enr", "cr"):
                self.assertTrue((self.tmp / tag / v / "d1.json").exists())
        self.assertTrue(all(c["spec"] == "ollama:qwen2.5:7b" for c in f_local.calls))
        self.assertTrue(all(c["spec"].startswith("openrouter:openai/gpt-oss-120b@deepinfra") for c in f_strong.calls))
        self.assertTrue(all(c["reasoning"] == "low" for c in f_strong.calls))
        self.assertTrue(all(c["seed"] == 0 for c in f_local.calls + f_strong.calls))
        st = json.loads((self.tmp / "strong" / "enr" / "d1.json").read_text())["stats"]
        self.assertTrue(st["spec"].startswith("openrouter:"))
        # resumable: a rebuild makes no new calls
        n = len(f_local.calls)
        self.builder("local", ["enr", "cr"], f_local).build_doc(doc)
        self.assertEqual(len(f_local.calls), n)
        self.assertEqual(build_v2.missing(["d1", "d2"], ["struct", "enr"], "strong", self.tmp), ["d2"])

    def test_parse_failures_retry_then_keep_empty(self):
        fake = FakeComplete(bad_json=4)                    # first chunk fails 1 + 3 retries
        self.builder("local", ["struct", "enr"], fake).build_doc({"id": "d1", "markdown": MD})
        d = json.loads((self.tmp / "local" / "enr" / "d1.json").read_text())
        n_struct = len(json.loads((self.tmp / "base" / "struct" / "d1.json").read_text())["chunks"])
        self.assertEqual(len(d["chunks"]), n_struct)       # nothing dropped
        self.assertTrue(d["chunks"][0]["llm_failed"])
        self.assertEqual(d["chunks"][0]["title"], "")
        self.assertFalse(any(c["llm_failed"] for c in d["chunks"][1:]))
        self.assertEqual(d["stats"]["failed_chunks"], 1)
        self.assertEqual([c["seed"] for c in fake.calls[:4]], [0, 1, 2, 3])

    def test_cr_document_chunks_consecutive_in_one_worker(self):
        docs = [{"id": f"d{i}", "markdown": MD.replace("Title", f"Doc{i}")} for i in range(6)]
        fake = FakeComplete(delay=0.002)
        b = self.builder("local", ["struct", "cr"], fake)
        res = b.run(docs, workers=3)
        self.assertEqual(res["done"], 6)
        seen_threads: dict = {}
        by_thread: dict = {}
        for c in fake.calls:
            doc = c["prompt"].split("# ", 1)[1].split("\n", 1)[0]
            seen_threads.setdefault(doc, set()).add(c["thread"])
            by_thread.setdefault(c["thread"], []).append((doc, c["prompt"]))
        self.assertTrue(all(len(t) == 1 for t in seen_threads.values()))   # one worker per document
        for seq in by_thread.values():                                      # contiguous runs per doc
            docs_in_order = [d for d, _ in seq]
            runs = [d for i, d in enumerate(docs_in_order) if i == 0 or d != docs_in_order[i - 1]]
            self.assertEqual(len(runs), len(set(runs)))
        for d in docs:                                                      # chunk order preserved
            got = json.loads((self.tmp / "local" / "cr" / f"{d['id']}.json").read_text())["chunks"]
            prompts = [p for t in by_thread.values() for doc, p in t if doc == d["markdown"].split("# ", 1)[1].split("\n")[0]]
            self.assertEqual([p.split("<chunk>\n")[1].split("\n</chunk>")[0] for p in prompts],
                             [c["text"] for c in got])
            self.assertTrue(all(c["context"].startswith("ctx for") for c in got))

    def test_budget_stop_is_clean(self):
        md = "\n\n".join(f"## Section {i}\n\n" + "text " * 250 for i in range(4))   # several chunks
        docs = [{"id": f"d{i}", "markdown": md} for i in range(5)]
        fake = FakeComplete(budget_after=2)
        res = self.builder("strong", ["struct", "enr"], fake).run(docs, workers=1)
        self.assertTrue(res["budget_stop"])
        self.assertEqual(res["failed"], [])
        self.assertFalse((self.tmp / "strong" / "enr" / "d0.json").exists())   # partial doc never cached
        self.assertEqual(len(fake.calls), 3)                                  # nothing after the stop
        self.assertEqual(build_v2.missing([d["id"] for d in docs], ["enr"], "strong", self.tmp),
                         [d["id"] for d in docs])

    def test_transport_error_fails_doc_not_cached(self):
        def boom(*a, **k):
            raise llm_api.LLMError("ollama: transport")
        res = self.builder("local", ["struct", "enr"], boom).run([{"id": "d1", "markdown": MD}], workers=1)
        self.assertEqual(res["failed"], ["d1"])
        self.assertFalse((self.tmp / "local" / "enr" / "d1.json").exists())

    def test_configure_llm_api(self):
        old = (llm_api.OLLAMA_URL, llm_api.NUM_CTX)
        try:
            build_v2.configure_llm_api("freshstack")
            self.assertEqual((llm_api.OLLAMA_URL, llm_api.NUM_CTX), ("http://127.0.0.1:11435", 32768))
            build_v2.configure_llm_api("qasper")
            self.assertEqual(llm_api.NUM_CTX, 16384)
        finally:
            llm_api.OLLAMA_URL, llm_api.NUM_CTX = old


# --------------------------------------------------------------------------- frozen config

class TestFrozen(Tmp):
    def cfg(self, digest="sha256:abc"):
        with mock.patch.object(build_v2, "git_state", return_value={"commit": "c0", "dirty": False}):
            return build_v2.current_config(digest_fn=lambda m: digest)

    def test_contents(self):
        c = self.cfg()
        from build import CR_PROMPT
        from mdkeychunker.enricher import ENRICH_PROMPT
        self.assertEqual(c["prompts_sha256"]["enrich"], hashlib.sha256(ENRICH_PROMPT.encode()).hexdigest())
        self.assertEqual(c["prompts_sha256"]["cr"], hashlib.sha256(CR_PROMPT.encode()).hexdigest())
        self.assertIn("answer_qasper", c["prompts_sha256"])
        self.assertEqual(c["models"]["strong"]["providers"], ["deepinfra/bf16"])
        self.assertEqual(c["models"]["strong"]["reasoning"], "low")
        self.assertEqual(c["models"]["local"]["num_ctx"], {"qasper": 16384, "freshstack": 32768})
        self.assertFalse(c["models"]["local"]["think"])
        self.assertIn("mdkeychunker", c["git"])

    def test_write_then_match_then_refuse(self):
        p = self.tmp / "frozen.json"
        build_v2.check_frozen(self.cfg(), p)
        self.assertTrue(p.exists())
        build_v2.check_frozen(self.cfg(), p)                                  # identical: ok
        c = self.cfg()
        c["git"]["harness"]["dirty"] = True
        build_v2.check_frozen(c, p)                                           # dirty flag is informational
        build_v2.check_frozen(self.cfg(digest=None), p)                       # unreachable ollama: kept
        bad = self.cfg()
        bad["prompts_sha256"]["cr"] = "0" * 64
        with self.assertRaises(build_v2.FrozenConfigMismatch) as e:
            build_v2.check_frozen(bad, p)
        self.assertIn("prompts_sha256.cr", str(e.exception))
        moved = self.cfg()
        moved["git"]["harness"]["commit"] = "c1"
        with self.assertRaises(build_v2.FrozenConfigMismatch):
            build_v2.check_frozen(moved, p)
        build_v2.check_frozen(moved, p, ignore_git=True)
        with self.assertRaises(build_v2.FrozenConfigMismatch):
            build_v2.check_frozen(self.cfg(), self.tmp / "absent.json", write=False)

    def test_main_refuses_before_any_work(self):
        p = self.tmp / "frozen.json"
        bad = self.cfg()
        bad["models"]["strong"]["spec"] = "openrouter:other@x"
        p.write_text(json.dumps(bad))
        with mock.patch.object(build_v2, "current_config", return_value=self.cfg()), \
             mock.patch.object(build_v2, "Builder") as B:
            rc = build_v2.main(["--dataset", "qasper", "--frozen", str(p)])
        self.assertEqual(rc, 2)
        B.assert_not_called()


# --------------------------------------------------------------------------- evaluation and stats

def write_caches(cache: Path, doc_id: str, md: str, systems: list[str]) -> None:
    from build import chunk_dict, fixed_chars, fixed_tokens
    from mdkeychunker.chunker import MarkdownChunker
    from mdkeychunker.config import Config
    struct = [chunk_dict(c) for c in MarkdownChunker(Config()).chunk(md)]
    sets = {"fixed512": fixed_chars(md), "fixedtok": fixed_tokens(md), "struct": struct,
            "enr": [{**c, "title": "T", "summary": "S", "keywords": ["k"], "questions": ["q"]} for c in struct],
            "cr": [{**c, "context": "context"} for c in struct]}
    for label in systems:
        tag, variant, _ = evaluate_v2.SYSTEMS_V2[label]
        p = build_v2.cache_path(tag, variant, doc_id, cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"chunks": sets[variant], "stats": {}}))


class TestEvaluate(Tmp):
    def test_rank_all_hybrid_matches_evaluate(self):
        docs = ["alpha beta gamma", "beta delta", "gamma epsilon zeta", "alpha alpha"]
        qs = ["alpha", "zeta gamma"]
        with mock.patch.object(evaluate, "embed", fake_embed):
            got = evaluate_v2.rank_all(docs, qs, ["bm25", "hybrid", "mxbai"])
            self.assertEqual([list(map(int, o)) for o in got["hybrid"]],
                             [list(map(int, o)) for o in evaluate.rank("hybrid", docs, qs)])
            self.assertEqual(set(got), {"bm25", "hybrid", "mxbai"})

    def test_plan_systems_paired_completeness(self):
        core = ["F512c", "F256t", "S", "S+T", "E", "CR"]
        for d in ("a", "b", "c"):
            write_caches(self.tmp, d, MD, core if d != "c" else core[:-1])   # c lacks CR
        write_caches(self.tmp, "a", MD, ["E-strong"])
        plan = evaluate_v2.plan_systems({"a": ["a"], "b": ["b"], "c": ["c"]}, self.tmp)
        self.assertEqual(plan["dropped"], ["c"])
        self.assertEqual(sorted(plan["groups"]), ["a", "b"])
        self.assertEqual(plan["strong_groups"], {"E-strong": ["a"]})

    def test_eval_qasper_and_freshstack_end_to_end(self):
        core = ["F512c", "F256t", "S", "S+T", "E", "CR"]
        papers = []
        for i in range(3):
            md = MD.replace("Title", f"Paper {i}")
            write_caches(self.tmp, f"p{i}", md, core)
            papers.append({"id": f"p{i}", "markdown": md, "questions": [
                {"qid": f"p{i}q0", "question": "Which retriever fuses rankings?",
                 "evidence": ["We use BM25 and dense embeddings. The hybrid retriever fuses both rankings."]},
                {"qid": f"p{i}q1", "question": "no evidence", "evidence": []}]})
        plan = evaluate_v2.plan_systems({p["id"]: [p["id"]] for p in papers}, self.tmp)
        with mock.patch.object(evaluate, "embed", fake_embed):
            res = evaluate_v2.eval_qasper(papers, plan, self.tmp, ["bm25", "hybrid"])
        self.assertEqual(res["question_ids"], {f"p{i}": [f"p{i}q0"] for i in range(3)})
        self.assertEqual(set(res["per"][("hybrid", "S", "rec@512t")]), {"p0", "p1", "p2"})
        self.assertEqual(res["per"][("hybrid", "S", "rec@1024t")]["p0"], [1.0])   # whole doc fits
        # FreshStack: pooled per domain, cluster = question
        for dom in ("laravel", "angular"):
            write_caches(self.tmp, f"{dom}__x.md", MD, core)
        dqs = {dom: [{"qid": f"{dom}-1", "question": "hybrid retriever",
                      "evidence": [{"text": "We use BM25 and dense embeddings. The hybrid retriever fuses both."}]}]
               for dom in ("laravel", "angular")}
        plan = evaluate_v2.plan_systems({d: [f"{d}__x.md"] for d in dqs}, self.tmp)
        with mock.patch.object(evaluate, "embed", fake_embed):
            fres = evaluate_v2.eval_freshstack(dqs, plan, self.tmp, ["hybrid"])
        self.assertEqual(fres["strata"], {"laravel-1": "laravel", "angular-1": "angular"})
        self.assertIn(("hybrid", "E", "prec@1024t"), fres["per"])
        evaluate_v2.dump_per_question(self.tmp / "out" / "freshstack_per_question.json", fres)
        back = evaluate_v2.load_per_question(self.tmp / "out" / "freshstack_per_question.json")
        self.assertEqual(back["strata"], fres["strata"])
        self.assertEqual(back["per"][("hybrid", "E", "mrr@10")], fres["per"][("hybrid", "E", "mrr@10")])

    def _synthetic(self, out: Path) -> None:
        rng = np.random.default_rng(1)
        # Qasper: 60 papers x 3 questions. E == S+T + tiny noise (equivalent), CR far below E,
        # S far above F512c, S == F256t.
        q = {}
        base = {p: list(rng.uniform(0.3, 0.7, 3)) for p in range(60)}
        def shift(delta, noise=0.01):
            return {f"p{p}": [float(np.clip(v + delta + rng.normal(0, noise), 0, 1)) for v in vs]
                    for p, vs in base.items()}
        for metric in ("rec@512t", "hit@5"):
            q[("hybrid", "S+T", metric)] = shift(0.0)
            q[("hybrid", "E", metric)] = shift(0.0)
            q[("hybrid", "CR", metric)] = shift(-0.2)
            q[("hybrid", "S", metric)] = shift(0.0)
            q[("hybrid", "F512c", metric)] = shift(-0.15)
            q[("hybrid", "F256t", metric)] = shift(0.0, 0.2)
            q[("bm25", "S", metric)] = shift(0.0)
            q[("bm25", "E", metric)] = shift(0.05)
        evaluate_v2.dump_per_question(out / "qasper_per_question.json",
                                      {"per": q, "question_ids": {f"p{p}": ["a", "b", "c"] for p in range(60)}})
        f = {}
        strata = {f"q{i}": ("laravel", "angular", "yolo")[i % 3] for i in range(90)}
        b2 = {k: [float(rng.uniform(0.2, 0.6))] for k in strata}
        for label, d in (("S+T", 0), ("E", 0.0), ("CR", 0.0), ("S", 0), ("F512c", -0.1), ("F256t", -0.1)):
            f[("hybrid", label, "prec@1024t")] = {k: [v[0] + d + float(rng.normal(0, 0.005))] for k, v in b2.items()}
        evaluate_v2.dump_per_question(out / "freshstack_per_question.json",
                                      {"per": f, "question_ids": {k: [k] for k in strata}, "strata": strata})
        e2e = {("hybrid", s, "answer_f1"): shift(0.0) for s in ("E", "S+T", "CR")}
        evaluate_v2.dump_per_question(out / "qasper_e2e_per_question.json", {"per": e2e, "question_ids": {}})

    def test_stats_families_on_synthetic_data(self):
        out = self.tmp / "res"
        self._synthetic(out)
        r = evaluate_v2.analyze(out, n=500)
        prim = {(x["dataset"], x["a"], x["b"]): x for x in r["primary"]}
        self.assertEqual(len(prim), 4)
        self.assertTrue(all(x["family_m"] == 4 and "p_holm" in x for x in prim.values()))
        self.assertEqual(prim[("qasper", "E", "S+T")]["verdict"], "equivalent")
        self.assertEqual(prim[("qasper", "E", "CR")]["verdict"], "different")
        self.assertEqual(prim[("qasper", "E", "S+T")]["margin"], 0.05)
        self.assertEqual(prim[("freshstack", "E", "CR")]["margin"], 0.04)
        self.assertEqual(prim[("freshstack", "E", "S+T")]["verdict"], "equivalent")
        self.assertLess(prim[("freshstack", "E", "S+T")]["ci90_bca"][1], 0.04)
        ps = [prim[k]["p_tost"] for k in prim]
        self.assertEqual([prim[k]["p_holm"] for k in prim], __import__("stats").holm(ps)["p_adj"])
        h3 = {(x["dataset"], x["a"], x["b"]): x for x in r["h3"]}
        self.assertTrue(h3[("qasper", "S", "F512c")]["reject_holm"])
        self.assertGreater(h3[("qasper", "S", "F512c")]["diff"], 0.1)
        self.assertNotIn("p_tost", h3[("qasper", "S", "F512c")])
        ks = r["key_secondary"]
        self.assertTrue(all(x["status"] == "not_available" for x in ks if "strong" in x["a"]))
        f1 = [x for x in ks if x["metric"] == "answer_f1" and x["status"] == "ok"]
        self.assertEqual(len(f1), 2)
        self.assertTrue(all(x["family_m"] == 2 for x in f1))
        ex = r["exploratory"]
        self.assertTrue(any(x["retriever"] == "bm25" and x["a"] == "E" for x in ex))
        self.assertFalse(any("p_holm" in x for x in ex))
        self.assertTrue(any(x["dataset"] == "freshstack/laravel" for x in ex))
        self.assertTrue((out / "stats.json").exists())
        self.assertIn("PRIMARY family", (out / "summary.txt").read_text())

    def test_freshstack_bootstrap_is_stratified(self):
        pa = {f"q{i}": [float(i % 3)] for i in range(30)}
        pb = {k: [0.0] for k in pa}
        strata = {f"q{i}": i % 3 for i in range(30)}
        a = evaluate_v2.compare(pa, pb, strata=strata, n=300)
        # with strata = the value classes, every resample keeps the class mix: zero width
        self.assertAlmostEqual(a["ci95_pct"][0], a["ci95_pct"][1])
        b = evaluate_v2.compare(pa, pb, n=300)
        self.assertGreater(b["ci95_pct"][1] - b["ci95_pct"][0], 0.1)


# --------------------------------------------------------------------------- e2e

class TestE2E(Tmp):
    def test_f1_wiring_and_resume(self):
        core = ["F512c", "F256t", "S", "S+T", "E", "CR"]
        papers = []
        for i in range(2):
            write_caches(self.tmp / "cache", f"p{i}", MD, core)
            papers.append({"id": f"p{i}", "questions": [
                {"qid": f"p{i}q0", "question": "Which retriever?", "evidence": ["hybrid retriever"],
                 "answers": [
                     {"unanswerable": False, "extractive_spans": ["the hybrid retriever"], "free_form_answer": "",
                      "yes_no": None},
                     {"unanswerable": False, "extractive_spans": [], "free_form_answer": "BM25 plus dense",
                      "yes_no": None}]},
                {"qid": f"p{i}q1", "question": "Is it good?", "evidence": ["improves recall"],
                 "answers": [{"unanswerable": False, "extractive_spans": [], "free_form_answer": "",
                              "yes_no": True}]}]})
        calls = []

        def gen(spec, prompt, **kw):
            calls.append((spec, prompt, kw))
            ans = "Yes" if "Is it good?" in prompt else "hybrid retriever"
            return {"text": f"Answer: {ans}", "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0}

        plan = evaluate_v2.plan_systems({p["id"]: [p["id"]] for p in papers}, self.tmp / "cache")
        with mock.patch.object(evaluate, "embed", fake_embed):
            res = e2e_v2.run(papers, plan, self.tmp / "cache", e2e_cache=self.tmp / "e2e", workers=2,
                             complete_fn=gen, log=lambda s: None)
        self.assertEqual(len(calls), 2 * 2 * 6)
        spec, prompt, kw = calls[0]
        self.assertEqual(spec, "ollama:qwen2.5:7b")
        self.assertEqual((kw["max_tokens"], kw["temperature"], kw["seed"]), (64, 0.0, 0))
        self.assertEqual(set(k[1] for k in res["per"]), set(core))
        self.assertEqual(res["per"][("hybrid", "S", "answer_f1")]["p0"], [1.0, 1.0])  # max over annotators
        self.assertTrue(all(r["context_tokens"] <= 512 for r in res["rows"]))
        row = next(r for r in res["rows"] if r["qid"] == "p0q0")
        self.assertEqual(row["answer"], "hybrid retriever")
        with mock.patch.object(evaluate, "embed", fake_embed):
            res2 = e2e_v2.run(papers, plan, self.tmp / "cache", e2e_cache=self.tmp / "e2e", workers=2,
                              complete_fn=gen, log=lambda s: None)
        self.assertEqual(len(calls), 24)                                            # all cached
        self.assertEqual(res2["per"], res["per"])

    def test_context_is_exactly_512_tokens(self):
        long_md = " ".join(f"word{i}" for i in range(3000))
        write_caches(self.tmp, "p", long_md, ["S"])
        got = {}

        def gen(spec, prompt, **kw):
            got["prompt"] = prompt
            return {"text": "x"}
        paper = {"id": "p", "questions": [{"qid": "q", "question": "word5?", "evidence": ["word5"],
                                           "answers": [{"unanswerable": True}]}]}
        with mock.patch.object(evaluate, "embed", fake_embed):
            rows = e2e_v2.answer_paper(paper, "S", self.tmp, e2e_cache=self.tmp / "e2e", complete_fn=gen)
        self.assertEqual(rows[0]["context_tokens"], 512)


# --------------------------------------------------------------------------- orchestrator

class TestOrchestrator(Tmp):
    def test_stage_skips_and_reductions(self):
        st = orchestrate_v2.stages(0)
        names = [s["name"] for s in st]
        self.assertEqual(names[0], "select")
        self.assertEqual(names[-2:], ["evaluate", "e2e"])
        self.assertTrue(orchestrate_v2.skipped(st[3], {"strong"}, set()))
        self.assertFalse(orchestrate_v2.skipped(st[1], {"strong"}, set()))
        self.assertTrue(orchestrate_v2.skipped(st[1], set(), {"evaluate"}))
        self.assertNotIn("strong-freshstack-cr", names)  # plan sec. 2: no CR-strong on FreshStack
        r1 = {s["name"]: s for s in orchestrate_v2.stages(1)}
        self.assertIn("--sample", r1["strong-qasper-cr"]["build"])
        self.assertNotIn("--domains", r1["strong-freshstack-enr"]["build"])
        r2 = {s["name"]: s for s in orchestrate_v2.stages(2)}
        self.assertEqual(r2["strong-freshstack-enr"]["build"][-2:], ["--domains", "laravel"])
        self.assertTrue(r2["local-qasper"]["ollama"])
        self.assertFalse(r2["strong-qasper-cr"]["ollama"])

    def test_build_targets_match_build_v2_sampling(self):
        sel = {"qasper": {"paper_ids": [f"p{i}" for i in range(300)]},
               "freshstack": {d: {"doc_ids": [f"{d}{i}" for i in range(3)]} for d in ("laravel", "angular", "yolo")}}
        ids, variants, tag = orchestrate_v2.build_targets(
            ["--dataset", "qasper", "--model-tag", "strong", "--variants", "cr", "--sample", "150"], sel)
        docs = [{"id": i} for i in sel["qasper"]["paper_ids"]]
        import random
        want = sorted(d["id"] for d in random.Random(2027).sample(docs, 150))
        self.assertEqual(ids, want)
        self.assertEqual((variants, tag), (["cr"], "strong"))
        ids, variants, _ = orchestrate_v2.build_targets(
            ["--dataset", "freshstack", "--variants", "enr", "--domains", "laravel", "--parapack"], sel)
        self.assertEqual(ids, ["laravel0", "laravel1", "laravel2"])
        self.assertEqual(variants, ["enr", "parapack"])

    def test_spend_monitor_logs_ledger_total(self):
        ledger = self.tmp / "ledger.jsonl"
        ledger.write_text(json.dumps({"cost_usd": 0.10}) + "\n" + json.dumps({"cost_usd": 0.02}) + "\n")
        events = self.tmp / "events.log"
        with mock.patch.object(orchestrate_v2, "EVENTS", events):
            m = orchestrate_v2.SpendMonitor(ledger)
            self.assertAlmostEqual(m.check(), 0.12)
            m.check()                                   # unchanged: no new line
            with ledger.open("a") as f:
                f.write(json.dumps({"cost_usd": 0.06}) + "\n")
            m.check()
        lines = events.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("SPEND $0.1800", lines[-1])

    def test_estimate_runs_on_selection(self):
        p = ROOT / "data" / "v2" / "selection.json"
        if not p.exists():
            self.skipTest("selection.json not built")
        e = orchestrate_v2.estimate(json.loads(p.read_text()))
        self.assertGreater(e["qasper"]["strong_cr_in_tokens"], 40e6)
        self.assertLess(orchestrate_v2.estimate(json.loads(p.read_text()), 2)["qasper"]["strong_cr_usd"],
                        e["qasper"]["strong_cr_usd"])


if __name__ == "__main__":
    unittest.main()
