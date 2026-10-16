"""End-to-end answers from a fixed token budget of retrieved source text.

build_context() fills exactly B cl100k tokens of chunk *text* (never generated
metadata) in rank order, cutting the last chunk at the budget. It is the same
selection as evaluate.retrieved_units(order, chunks, budget=B), so answer-level
and retrieval-level metrics see identical evidence. Chunk separators and the
"[n]" markers are not counted toward B.

The Qasper answer metric is a port of the official evaluator, the
qasper-led-baseline script:
  https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
  (raw: https://raw.githubusercontent.com/allenai/qasper-led-baseline/main/scripts/evaluator.py,
   fetched 2026-10-06)
normalize_answer and token_f1_score are copied verbatim (SQuAD v1.1 rules).
qasper_references() mirrors get_answers_and_evidence(): an unanswerable
annotation becomes the reference "Unanswerable"; extractive spans are joined
with ", "; then free-form; then yes_no True -> "Yes", False -> "No". The score of
a prediction is the maximum token F1 over references, and its answer type is
the type of the first best-scoring reference (the script's stable sort).
"""
from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from typing import Any, Iterable

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")  # same encoder as evaluate.py

QASPER_PROMPT = """You are answering a question about a scientific paper using only the excerpts below.

Excerpts:
{context}

Question: {question}

Instructions:
- Answer with the shortest span of words from the excerpts that answers the question, or a short phrase if no span fits.
- If it is a yes/no question, answer only "Yes" or "No".
- If the excerpts do not contain the answer, answer only "Unanswerable".
- At most 30 words. No explanation, no preamble.

Answer:"""

FRESHSTACK_PROMPT = """You are answering a Stack Overflow question using the documentation excerpts below.

Documentation excerpts:
{context}

Question:
{question}

Write a concise, correct answer to the question. Use the excerpts where they help; include short code only when it is needed. At most 250 words.

Answer:"""

PROMPTS = {"qasper": QASPER_PROMPT, "freshstack": FRESHSTACK_PROMPT}
# Defaults frozen by the analysis plan: deterministic decoding, capped length.
GEN_SETTINGS = {
    "qasper": {"max_tokens": 64, "temperature": 0.0, "seed": 0},
    "freshstack": {"max_tokens": 512, "temperature": 0.0, "seed": 0},
}


def prompt_hashes() -> dict[str, str]:
    """sha256 of each prompt template, for the frozen-by-hash record."""
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in PROMPTS.items()}


# --------------------------------------------------------------------------- context

def select_text(order: Iterable[int], chunks: list[dict], budget: int) -> list[str]:
    """Chunk texts in rank order, total exactly `budget` cl100k tokens (fewer only
    if the corpus is shorter). Mirrors evaluate.retrieved_units(..., budget=B)."""
    out, used = [], 0
    for i in order:
        if used >= budget:
            break
        toks = ENC.encode(chunks[i]["text"])[: budget - used]
        used += len(toks)
        out.append(ENC.decode(toks))
    return out


def build_context(order: Iterable[int], chunks: list[dict], budget: int) -> tuple[str, int]:
    """Numbered context string and the number of source-text tokens it holds."""
    pieces = select_text(order, chunks, budget)
    n = sum(len(ENC.encode(p)) for p in pieces)
    return "\n\n".join(f"[{k}] {p.strip()}" for k, p in enumerate(pieces, 1)), n


def build_prompt(dataset: str, question: str, context: str) -> str:
    return PROMPTS[dataset].format(context=context, question=question.strip())


_UNANS = {"unanswerable", "unanswerable question", "question is unanswerable", "not answerable",
          "cannot be answered", "cannot answer", "no answer", "not mentioned", "not stated",
          "not specified", "not provided", "not given", "excerpts do not contain answer"}


def clean_answer(text: str, dataset: str) -> str:
    """Strip preambles; map refusal variants to the reference string "Unanswerable"."""
    s = (text or "").strip()
    s = re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", s, flags=re.I).strip()
    if dataset == "qasper":
        s = s.splitlines()[0].strip() if s else s
        if normalize_answer(s) in _UNANS:
            return "Unanswerable"
    return s


def generate_answer(dataset: str, question: str, order: Iterable[int], chunks: list[dict], budget: int,
                    model_spec: str, *, complete_fn=None, reasoning: Any = None, tag: str = "") -> dict:
    """Generate one answer. Returns the answer plus everything needed for the log row."""
    if complete_fn is None:
        from llm_api import complete as complete_fn
    context, n_tok = build_context(order, chunks, budget)
    prompt = build_prompt(dataset, question, context)
    r = complete_fn(model_spec, prompt, json_mode=False, reasoning=reasoning, tag=tag or f"answer-{dataset}",
                    **GEN_SETTINGS[dataset])
    return {"answer": clean_answer(r["text"], dataset), "raw": r["text"], "context_tokens": n_tok,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            **{k: r.get(k) for k in ("in_tokens", "out_tokens", "cost_usd", "provider", "model",
                                     "latency_s", "finish_reason")}}


# --------------------------------------------------------------------------- official Qasper F1

def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _annotation_list(answers: Any) -> list[dict]:
    """Accept the official JSON shape (list of {"answer": {...}}) or the HF parquet
    shape ({"answer": [...], "annotation_id": [...], ...})."""
    if isinstance(answers, dict) and isinstance(answers.get("answer"), list):
        return list(answers["answer"])
    return [a["answer"] if "answer" in a else a for a in answers]


def qasper_references(answers: Any) -> list[dict]:
    """References for one question, exactly as get_answers_and_evidence() builds them."""
    refs = []
    for info in _annotation_list(answers):
        if info["unanswerable"]:
            refs.append({"answer": "Unanswerable", "type": "none"})
        elif info["extractive_spans"] is not None and len(info["extractive_spans"]) > 0:
            refs.append({"answer": ", ".join(info["extractive_spans"]), "type": "extractive"})
        elif info["free_form_answer"]:
            refs.append({"answer": info["free_form_answer"], "type": "abstractive"})
        elif info["yes_no"]:
            refs.append({"answer": "Yes", "type": "boolean"})
        elif info["yes_no"] is not None:
            refs.append({"answer": "No", "type": "boolean"})
        else:
            raise RuntimeError("annotation does not contain an answer")
    return refs


def qasper_answer_f1(prediction: str, references: list[dict]) -> tuple[float, str]:
    """(max token F1 over references, type of the best reference), as evaluate()."""
    scored = [(token_f1_score(prediction, r["answer"]), r["type"]) for r in references]
    f1, typ = sorted(scored, key=lambda x: x[0], reverse=True)[0]
    return float(f1), typ
