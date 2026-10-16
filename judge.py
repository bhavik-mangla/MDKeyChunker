"""Nugget-based answer judging for FreshStack.

Follows the FreshStack evaluation (Thakur et al., 2025, arXiv:2504.13128), which
scores RAG answers by nugget recall using the TREC 2024 RAG AutoNuggetizer
support labels (support / partial_support / no_support; Pradeep et al., 2024,
arXiv:2411.09607). Strict recall counts only "support"; partial recall counts
partial support as one half. One judge call labels all nuggets of an answer.

The judge must come from a different model family than the generator and the
enrichment models, so it never grades text written by its own family.
"""
from __future__ import annotations

import json
import re
from typing import Callable

import llm_api

LABELS = ("support", "partial_support", "no_support")
_ALIASES = {"support": "support", "supported": "support", "full_support": "support",
            "partial_support": "partial_support", "partial": "partial_support",
            "partially_supported": "partial_support",
            "no_support": "no_support", "not_support": "no_support", "not_supported": "no_support",
            "unsupported": "no_support", "none": "no_support"}
MAX_PARSE_ATTEMPTS = 3
POSITIVE_MIN = 0.85
NEGATIVE_MAX = 0.05

PROMPT = """You are judging whether an answer to a technical question supports a list of reference facts ("nuggets").

For each nugget, decide from the answer text alone:
- "support": the answer fully states or clearly entails the nugget.
- "partial_support": the answer states part of the nugget or states it only vaguely.
- "no_support": the answer does not state the nugget, or contradicts it.
Do not use outside knowledge. Do not reward length or style.

Question:
{question}

Answer:
{answer}

Nuggets:
{nuggets}

Return only JSON of the form {{"labels": [{{"id": 1, "label": "support"}}, ...]}} with exactly one entry per nugget id."""


class JudgeParseError(ValueError):
    pass


def _norm_label(x: object) -> str:
    key = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIASES:
        raise JudgeParseError(f"unknown label {x!r}")
    return _ALIASES[key]


def parse_nugget_labels(text: str, n: int) -> list[str]:
    """Labels in nugget order, or JudgeParseError. Accepts {"labels": [ {id,label}... ]} or a plain list."""
    try:
        data = llm_api.extract_json(text)
    except Exception as e:  # noqa: BLE001 - any failure to find JSON is a parse error
        raise JudgeParseError(f"no JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("labels"), list):
        raise JudgeParseError("missing 'labels' list")
    items = data["labels"]
    if len(items) != n:
        raise JudgeParseError(f"expected {n} labels, got {len(items)}")
    if all(isinstance(it, str) for it in items):
        return [_norm_label(it) for it in items]
    out: dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict) or "id" not in it or "label" not in it:
            raise JudgeParseError(f"bad item {it!r}")
        try:
            i = int(it["id"])
        except (TypeError, ValueError) as e:
            raise JudgeParseError(f"bad id {it['id']!r}") from e
        if not 1 <= i <= n or i in out:
            raise JudgeParseError(f"id {i} out of range or repeated")
        out[i] = _norm_label(it["label"])
    return [out[i] for i in range(1, n + 1)]


def nugget_scores(labels: list[str]) -> dict:
    n = len(labels)
    s = sum(lab == "support" for lab in labels)
    p = sum(lab == "partial_support" for lab in labels)
    return {"strict": s / n if n else 0.0, "partial": (s + 0.5 * p) / n if n else 0.0}


class Judge:
    def __init__(self, model_spec: str, *, other_models: list[str] | None = None,
                 complete_fn: Callable | None = None, max_tokens: int = 1024):
        fam = llm_api.model_family(model_spec)
        clash = [m for m in (other_models or []) if llm_api.model_family(m) == fam]
        if clash:
            raise ValueError(f"judge family {fam!r} matches generator/enricher models {clash}")
        self.model_spec = model_spec
        self.complete = complete_fn or llm_api.complete
        self.max_tokens = max_tokens
        self.parse_failures = 0
        self.calls = 0

    def judge_nuggets(self, question: str, answer: str, nuggets: list[str]) -> dict:
        prompt = PROMPT.format(question=question.strip(), answer=answer.strip(),
                               nuggets="\n".join(f"{i}. {n.strip()}" for i, n in enumerate(nuggets, 1)))
        last: Exception | None = None
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            self.calls += 1
            out = self.complete(self.model_spec, prompt, json_mode=True, max_tokens=self.max_tokens,
                                temperature=0.0, seed=attempt - 1, tag="judge")
            try:
                labels = parse_nugget_labels(out["text"], len(nuggets))
            except JudgeParseError as e:
                self.parse_failures += 1
                last = e
                continue
            return {"labels": labels, **nugget_scores(labels), "parse_attempts": attempt}
        raise JudgeParseError(f"unparseable after {MAX_PARSE_ATTEMPTS} attempts: {last}")


def freshstack_controls(judge: Judge, items: list[dict]) -> dict:
    """Positive control: a question's own gold answer. Negative control: another question's gold answer."""
    pos, neg, pairs = [], [], []
    n = len(items)
    for i, it in enumerate(items):
        pos.append(judge.judge_nuggets(it["question"], it["gold_answer"], it["nuggets"])["strict"])
        other = items[(i + 1) % n]
        if n > 1:
            pairs.append((it["qid"], other["qid"]))
            neg.append(judge.judge_nuggets(it["question"], other["gold_answer"], it["nuggets"])["strict"])
    p = sum(pos) / len(pos) if pos else 0.0
    q = sum(neg) / len(neg) if neg else 0.0
    return {"positive_strict_recall": p, "negative_strict_recall": q,
            "positive_pass": p >= POSITIVE_MIN, "negative_pass": q <= NEGATIVE_MAX,
            "negative_pairs": pairs, "n": n}


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def length_bias(rows: list[dict]) -> dict:
    """Per system: Spearman correlation between judge score and answer length in words."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["system"], []).append(r)
    return {s: {"n": len(rs), "spearman_score_length": _spearman([r["score"] for r in rs],
                                                                 [float(r["answer_words"]) for r in rs])}
            for s, rs in by.items()}


def answer_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


if __name__ == "__main__":  # tiny self-check of the parser
    print(parse_nugget_labels(json.dumps({"labels": [{"id": 1, "label": "support"}]}), 1))
