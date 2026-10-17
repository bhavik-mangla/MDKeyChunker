"""evidence_recall / evidence_precision: matching rules ported from the study, budget, offsets."""
import pytest

from mdkeychunker.evaluate import (
    CharTokenizer,
    evidence_precision,
    evidence_recall,
    get_tokenizer,
    to_gold_spans,
)
from mdkeychunker.models import Chunk, GoldSpan, Hit

TEN = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"


class WordTok:
    """One token per whitespace-separated word, so budgets in tests are easy to read."""

    def encode(self, text):
        self.ws = text.split()
        return list(range(len(self.ws)))

    def decode(self, tokens):
        return " ".join(self.ws[: len(tokens)])


def mk_chunk(text, cid="", **kw):
    return Chunk(text=text, chunk_id=cid or text[:12], **kw)


# --------------------------------------------------------------------------- text matching

def test_span_fully_inside_one_chunk():
    assert evidence_recall([mk_chunk("intro. " + TEN + " outro.")], [TEN]) == 1.0


def test_span_partially_in_chunk_uses_half_of_5grams():
    # 10 words -> 6 five-grams. Chunk holds w0..w7: grams starting w0..w3 = 4/6 >= 0.5
    assert evidence_recall([mk_chunk("w0 w1 w2 w3 w4 w5 w6 w7")], [TEN]) == 1.0
    # w0..w6: 3/6 = 0.5 -> covered (threshold is inclusive)
    assert evidence_recall([mk_chunk("w0 w1 w2 w3 w4 w5 w6")], [TEN]) == 1.0
    # w0..w5: 2/6 < 0.5
    assert evidence_recall([mk_chunk("w0 w1 w2 w3 w4 w5")], [TEN]) == 0.0


def test_span_across_two_chunks_counts_grams_per_chunk_not_across_boundary():
    a, b = mk_chunk("w0 w1 w2 w3 w4 w5", "a"), mk_chunk("w6 w7 w8 w9", "b")
    # a gives 2 grams (w0, w1 starts); b has < 5 words so its unit gram is the 4-tuple; 2/6 < 0.5
    assert evidence_recall([a, b], [TEN]) == 0.0
    a2, b2 = mk_chunk("w0 w1 w2 w3 w4 w5 w6", "a2"), mk_chunk("w3 w4 w5 w6 w7 w8 w9", "b2")
    assert evidence_recall([a2, b2], [TEN]) == 1.0  # together all 6 grams


def test_short_evidence_needs_contiguous_words_in_one_chunk():
    assert evidence_recall([mk_chunk("the answer is forty two")], ["forty two"]) == 1.0
    assert evidence_recall([mk_chunk("forty"), mk_chunk("two")], ["forty two"]) == 0.0
    assert evidence_recall([mk_chunk("two forty")], ["forty two"]) == 0.0


def test_case_and_punctuation_are_ignored():
    assert evidence_recall([mk_chunk("W0, W1; W2 - w3 (w4) w5 w6 w7 w8 w9!")], [TEN]) == 1.0


def test_recall_is_fraction_of_spans():
    gold = [TEN, "never retrieved words here at all"]
    assert evidence_recall([mk_chunk(TEN)], gold) == 0.5


def test_embed_text_is_not_used_for_matching():
    c = Chunk(text="body only", embed_text="Guide > Install\n\nbody only", chunk_id="x")
    assert evidence_recall([c], ["guide install"]) == 0.0


# --------------------------------------------------------------------------- budget

def test_budget_cut_mid_chunk_counts_the_kept_part():
    filler = mk_chunk("f0 f1 f2 f3 f4 f5", "fill")
    gold_chunk = mk_chunk(TEN, "gold")
    tok = WordTok()
    # 6 filler tokens + 7 of the gold chunk: w0..w6 kept -> 3/6 grams -> covered
    assert evidence_recall([filler, gold_chunk], [TEN], budget_tokens=13, tokenizer=tok) == 1.0
    # 6 + 5: w0..w4 -> 1/6
    assert evidence_recall([filler, gold_chunk], [TEN], budget_tokens=11, tokenizer=tok) == 0.0
    # budget used up before the gold chunk
    assert evidence_recall([filler, gold_chunk], [TEN], budget_tokens=6, tokenizer=tok) == 0.0
    assert evidence_recall([filler, gold_chunk], [TEN], budget_tokens=None, tokenizer=tok) == 1.0


def test_zero_budget_reads_nothing():
    assert evidence_recall([mk_chunk(TEN)], [TEN], budget_tokens=0) == 0.0
    with pytest.raises(ValueError):
        evidence_recall([mk_chunk(TEN)], [TEN], budget_tokens=-1)


def test_duplicate_retrieval_counts_once_and_uses_budget_once():
    filler = mk_chunk("f0 f1 f2 f3 f4 f5", "fill")
    gold_chunk = mk_chunk(TEN, "gold")
    tok = WordTok()
    # without dedupe the second filler would eat the budget before the gold chunk
    got = evidence_recall([filler, filler, gold_chunk], [TEN], budget_tokens=16, tokenizer=tok)
    assert got == 1.0


def test_k_limits_results_before_budget():
    assert evidence_recall([mk_chunk("x y z", "x"), mk_chunk(TEN, "t")], [TEN], k=1) == 0.0
    assert evidence_recall([mk_chunk("x y z", "x"), mk_chunk(TEN, "t")], [TEN], k=2) == 1.0


def test_default_tokenizer_is_chars_over_four():
    t = CharTokenizer()
    assert t.count("") == 0 and t.count("abcd") == 1 and t.count("abcde") == 2
    assert t.truncate("abcdefghij", 1) == "abcd" and t.truncate("ab", 5) == "ab"
    assert get_tokenizer(t) is t
    assert evidence_recall([mk_chunk(TEN)], [TEN], budget_tokens=4,
                           tokenizer=CharTokenizer(chars_per_token=8)) == 1.0
    assert isinstance(get_tokenizer(None), CharTokenizer)
    assert isinstance(get_tokenizer("estimate"), CharTokenizer)
    with pytest.raises(TypeError):
        get_tokenizer(42)  # type: ignore[arg-type]


def test_tiktoken_by_name():
    pytest.importorskip("tiktoken")
    enc = get_tokenizer("cl100k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    # the same spellings as mk.chunk
    for spec in ("tiktoken", "tiktoken:cl100k_base"):
        assert get_tokenizer(spec).name == "cl100k_base"
    assert get_tokenizer("tiktoken:o200k_base").name == "o200k_base"
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("hf:bert")


# --------------------------------------------------------------------------- result types

def test_hits_chunks_and_ids_are_interchangeable():
    c = mk_chunk(TEN, "t")
    assert evidence_recall([Hit(c, 1.0)], [TEN]) == 1.0
    assert evidence_recall(["t"], [TEN], chunks=[c]) == 1.0
    assert evidence_recall(["t"], [TEN], chunks={"t": c}) == 1.0
    with pytest.raises(ValueError, match="chunks="):
        evidence_recall(["t"], [TEN])
    with pytest.raises(KeyError):
        evidence_recall(["nope"], [TEN], chunks=[c])
    with pytest.raises(TypeError):
        evidence_recall([3.5], [TEN])  # type: ignore[list-item]


def test_empty_results_and_empty_gold():
    assert evidence_recall([], [TEN]) == 0.0
    with pytest.raises(ValueError):
        evidence_recall([mk_chunk(TEN)], [])


# --------------------------------------------------------------------------- doc ids and offsets

def test_text_gold_with_doc_id_only_matches_that_document():
    c = mk_chunk(TEN, "t", doc_id="a")
    assert evidence_recall([c], [GoldSpan(doc_id="a", text=TEN)]) == 1.0
    assert evidence_recall([c], [GoldSpan(doc_id="b", text=TEN)]) == 0.0
    assert evidence_recall([c], [GoldSpan(text=TEN)]) == 1.0  # no doc id -> any document


def _doc_chunks():
    doc = "aaaa bbbb cccc dddd"  # chunk boundaries at 0-10, 10-19
    return [Chunk(text=doc[0:10], doc_id="d", start_char=0, end_char=10, chunk_id="c0"),
            Chunk(text=doc[10:19], doc_id="d", start_char=10, end_char=19, chunk_id="c1")]


def test_offset_gold_needs_half_of_its_characters():
    c0, c1 = _doc_chunks()
    assert evidence_recall([c0], [("d", 2, 8)]) == 1.0
    assert evidence_recall([c0], [("d", 6, 14)]) == 1.0   # 4/8 inside c0
    assert evidence_recall([c0], [("d", 7, 15)]) == 0.0   # 3/8
    assert evidence_recall([c0, c1], [("d", 7, 15)]) == 1.0  # across two chunks
    assert evidence_recall([c0], [("other", 2, 8)]) == 0.0


def test_offset_gold_respects_budget_cut():
    c0, _ = _doc_chunks()
    # CharTokenizer: 1 token = 4 chars, so budget 1 keeps c0[0:4]
    assert evidence_recall([c0], [("d", 0, 4)], budget_tokens=1) == 1.0
    assert evidence_recall([c0], [("d", 5, 9)], budget_tokens=1) == 0.0


def test_offset_gold_without_chunk_offsets_is_an_error_not_zero():
    with pytest.raises(ValueError, match="offsets"):
        evidence_recall([mk_chunk(TEN)], [("d", 0, 5)])


def test_to_gold_spans_forms_and_validation():
    spans = to_gold_spans(["t", {"text": "u", "doc_id": "d"}, {"doc_id": "d", "start": 1, "end": 3},
                           ("d", 0, 2), GoldSpan(text="v")])
    assert [s.text for s in spans] == ["t", "u", "", "", "v"]
    assert spans[2].start == 1 and spans[3].end == 2
    for bad in (GoldSpan(start=1), GoldSpan(start=3, end=3), "  ", 7):
        with pytest.raises((ValueError, TypeError)):
            to_gold_spans([bad])  # type: ignore[list-item]


# --------------------------------------------------------------------------- precision

def test_precision_is_gold_share_of_retrieved_grams():
    gold = [TEN]
    assert evidence_precision([mk_chunk(TEN)], gold) == 1.0
    half = mk_chunk("w0 w1 w2 w3 w4 w5 x0 x1 x2 x3 x4", "h")  # 7 grams, 2 gold
    assert evidence_precision([half], gold) == pytest.approx(2 / 7)
    assert evidence_precision([], gold) == 0.0


def test_precision_with_offsets():
    c0, c1 = _doc_chunks()
    got = evidence_precision([c0, c1], [("d", 0, 10)], budget_tokens=None)
    assert got == pytest.approx(10 / 19)
    with pytest.raises(ValueError, match="mix"):
        evidence_precision([c0], [("d", 0, 4), "text"])
