"""Offline tests for the Amendment 2 merge arms and NoRollingKeysEnricher.

Run:  .venv/bin/python -m unittest tests/test_h4.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_h4  # noqa: E402
import build_v2  # noqa: E402
from mdkeychunker.models import Chunk  # noqa: E402


def _chunks(texts, keys):
    return [{"text": t, "section": "s", "title": f"t{i}", "summary": f"sum{i}", "keywords": [f"k{i}"],
             "questions": [f"q{i}"], "key": k, "related_keys": [], "entities": []}
            for i, (t, k) in enumerate(zip(texts, keys))]


class MergeArms(unittest.TestCase):
    def test_stage3_merges_same_key_and_keeps_all_text(self):
        e = _chunks(["alpha one", "beta two", "alpha three"], ["a", "b", "a"])
        m, orphans = build_h4.stage3(e)
        self.assertEqual((len(m), orphans), (2, 0))
        joined = " ".join(c["text"] for c in m)
        self.assertTrue(all(c["text"] in joined for c in e))

    def test_semantic_merge_picks_most_similar_adjacent_pair(self):
        vecs = np.array([[1, 0], [0, 1], [0.1, 0.99]], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        with mock.patch.object(build_h4.evaluate, "embed", return_value=vecs):
            out, done = build_h4.semantic_merge(_chunks(["a", "b", "c"], ["", "", ""]), 1)
        self.assertEqual(done, 1)
        self.assertEqual([c["text"] for c in out], ["a", "b\n\nc"])

    def test_semantic_merge_respects_cap(self):
        with mock.patch.object(build_h4.evaluate, "embed", return_value=np.eye(2, dtype=np.float32)):
            out, done = build_h4.semantic_merge(_chunks(["x" * 2000, "y" * 2000], ["", ""]), 1)
        self.assertEqual((done, len(out)), (0, 2))


class NoRollingKeys(unittest.TestCase):
    def test_prompt_never_shows_prior_keys(self):
        prompts = []

        class Fake:
            def call_json(self, prompt, max_tokens=1000):
                prompts.append(prompt)
                return {"key": "same key", "title": "t", "summary": "s"}

        out = build_v2.NoRollingKeysEnricher(Fake()).enrich_chunks([Chunk(text="a"), Chunk(text="b")])
        self.assertIn("(not provided)", prompts[1])
        self.assertNotIn("same key", prompts[1])
        self.assertEqual(out[1].key, "same key")


class PerField(unittest.TestCase):
    def test_field_instructions_are_verbatim_from_enrich_prompt(self):
        from mdkeychunker.enricher import ENRICH_PROMPT
        plain = ENRICH_PROMPT.replace("{{", "{").replace("}}", "}")
        for line in [*build_v2.PF_FIELDS.values(), *build_v2.PF_RULES.values()]:
            self.assertIn(line, plain)

    def test_four_calls_per_chunk_and_fields_filled(self):
        prompts = []

        class Fake:
            fatal = None
            failed = []

            def call_json(self, prompt, max_tokens=1000):
                prompts.append(prompt)
                for f in ("title", "summary", "keywords", "questions"):
                    if f'"{f}":' in prompt:
                        return {f: ["a", "b"] if f in ("keywords", "questions") else f"{f} text"}

            def stats(self):
                return {}

        out, _ = build_v2.perfield([Chunk(text="x", section_title="S"), Chunk(text="y")], Fake())
        self.assertEqual(len(prompts), 8)
        self.assertEqual(out[0]["title"], "title text")
        self.assertEqual(out[1]["questions"], ["a", "b"])
        self.assertIn("summary text", prompts[4])  # chunk 2 sees chunk 1's PF summary
        self.assertNotIn("Rolling Keys", prompts[0])
        self.assertTrue(prompts[2].rstrip().endswith("no extra text"))


class KeyExpansion(unittest.TestCase):
    def test_links_and_count_matched_adjacency(self):
        import evaluate_h5
        ch = [{"key": "a", "related_keys": []}, {"key": "b", "related_keys": []},
              {"key": "c", "related_keys": ["a"]}, {"key": "a", "related_keys": []},
              {"key": "a", "related_keys": []}]
        doc = ["d1", "d1", "d1", "d1", "d2"]
        lk = evaluate_h5.links(ch, doc)
        self.assertEqual(lk[0], [2, 3])          # c points to a; same key a; d2 excluded
        kx, ax = evaluate_h5.expand([0, 1], lk, doc)
        self.assertEqual(kx, [0, 2, 3, 1])
        self.assertEqual(ax, [0, 1, 2])           # 0 brings its 2 nearest same-doc neighbours; 1 brings 0
        kx, ax = evaluate_h5.expand([3], lk, doc)
        self.assertEqual((kx, ax), ([3, 0, 2], [3, 2, 1]))  # d2's chunk 4 is never a neighbour


class Consolidation(unittest.TestCase):
    def _adapter(self, reply):
        class Fake:
            fatal = None
            failed = []
            prompts = []

            def call_json(self, prompt, max_tokens=1000):
                self.prompts.append(prompt)
                return reply

            def stats(self):
                return {"calls": len(self.prompts)}
        return Fake()

    def test_remaps_only_to_listed_keys(self):
        ch = _chunks(["a", "b", "c"], ["oauth flow", "oauth token flow", "rate limits"])
        ad = self._adapter({"oauth token flow": "oauth flow", "rate limits": "made up key", "oauth flow": "oauth flow"})
        out, st = build_h4.consolidate(ch, ad)
        self.assertEqual([c["key"] for c in out], ["oauth flow", "oauth flow", "rate limits"])
        self.assertEqual((st["keys_before"], st["keys_after"], st["mapped"]), (3, 2, 1))
        self.assertIn("2-5 words", ad.prompts[0])

    def test_single_key_makes_no_call(self):
        ad = self._adapter({})
        out, st = build_h4.consolidate(_chunks(["a", "b"], ["k", "k"]), ad)
        self.assertEqual(ad.prompts, [])


class SectionExpansion(unittest.TestCase):
    def test_same_doc_same_section_only(self):
        import evaluate_h5
        ch = [{"section": "A"}, {"section": "B"}, {"section": "A"}, {"section": "A"}]
        self.assertEqual(evaluate_h5.section_expand([2, 1], ch, ["d", "d", "d", "e"]), [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
