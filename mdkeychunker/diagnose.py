"""``diagnose``: is your retriever leaving recall on the table, and would enrichment pay?

On your documents and a few labelled questions, ``diagnose`` runs reference retrievers (and
any of your own), scores each question by gold evidence found within a token budget
(``evidence_recall``, chunk-agnostic), and compares retrievers pairwise with the v4 study's
statistics: paired cluster bootstrap (BCa) CIs, a sign-flip test, Holm adjustment across
the report's comparisons, and the minimum detectable difference (MDE). When a CI includes
zero it says "inconclusive at n=..." instead of naming a winner.

Checks (``check=``):

- ``"fusion"``: hybrid (BM25 + dense, RRF) vs its dense half and its BM25 half.
- ``"embedder"``: a stronger dense embedder (``upgrade_embedder``) vs the baseline.
- ``"rerank"``: the baseline with a cross-encoder reranking its top 30 vs the baseline.
- ``"enrich"``: retrieval over LLM-enriched chunks (``enriched=`` or ``llm=``) vs plain
  chunks, for BM25 and hybrid.

The baseline is your retriever when you pass exactly one in ``retrievers``, else the
built-in hybrid. Built-in retrievers need ``pip install "mdkeychunker[retrieval]"``; with
only your own retrievers and ``check=()``, ``diagnose`` needs no optional dependency.

Example (toy embedder and reranker so nothing is downloaded):
    >>> from mdkeychunker.diagnose import diagnose
    >>> from mdkeychunker.retrieval import Reranker
    >>> docs = {"d": "# Install\\n\\nRun pip install foo on Linux.\\n\\n# Pricing\\n\\n"
    ...              "Plans start at ten dollars a month for teams."}
    >>> qa = [{"question": "how do I install it?", "gold": ["pip install foo"], "doc_id": "d"}]
    >>> report = diagnose(docs, qa, check=("rerank",), n_boot=200, bm25="study",
    ...                   embedder=lambda ts: [[float("install" in t.lower()), 1.0] for t in ts],
    ...                   reranker=Reranker(score_fn=lambda ps: [0.0 for _ in ps]))
    >>> sorted(report.arms)  # only arms that take part in a comparison are scored
    ['hybrid', 'hybrid+rerank']
    >>> report.findings[0].verdict  # a one-chunk-per-section toy: the reranker changes nothing
    'identical'
    >>> report.warnings[0][:47]
    'Your budget_tokens (512) covers whole documents'
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union

from . import stats
from .evaluate import Tokenizer, evidence_precision, evidence_recall, to_gold_spans
from .models import Chunk, GoldSpan, Hit, RetrievedItem, Retriever, SearchFn

__all__ = ["Finding", "QA", "Report", "diagnose"]

CHECKS = ("fusion", "embedder", "rerank", "enrich")
EmbedderLike = Union[str, Callable[[list[str]], Any]]

# Study context printed next to findings (never used as the user's thresholds).
_CONTEXT = {
    "fusion": "In the v4 study, untuned RRF of BM25 + nomic scored below its own dense half on "
              "Qasper (42.7 vs 47.8 recall@512, free prefix; exploratory).",
    "embedder": "In the v4 study, changing the retriever over free chunks gained more than LLM "
                "enrichment (+3.4 to +6.3 points; exploratory).",
    "rerank": "In the v4 study, a Qwen3 reranker over the hybrid top 30 was among the two "
              "strongest retrievers (Qasper free prefix 42.7 -> 54.5; exploratory).",
    "enrich": "In the v4 study (7B indexing LLM), enrichment helped most under BM25 (+8.7 on "
              "Qasper) and was equivalent to the free prefix under the strongest retrievers "
              "(secondary).",
}
_SCOPE = ("Study scope: Qasper (within-paper) and FreshStack, English, 7B/120B indexing LLMs, "
          "eight retrievers. Your data may differ; that is why diagnose measures it.")


# --------------------------------------------------------------------------- inputs

@dataclass
class QA:
    """One labelled question: its gold evidence and, optionally, the document it is about."""

    question: str
    gold: list[GoldSpan]
    doc_id: str = ""
    qid: str = ""


def _to_qa(items: Iterable[Any]) -> list[QA]:
    out = []
    for i, it in enumerate(items):
        if isinstance(it, QA):
            q = it
        elif isinstance(it, Mapping):
            text = it.get("question", it.get("query"))
            gold = it.get("gold", it.get("evidence"))
            if not isinstance(text, str) or gold is None:
                raise ValueError(f"qa item {i} needs 'question' and 'gold': {it!r}")
            gold_list = [gold] if isinstance(gold, (str, GoldSpan)) else list(gold)
            q = QA(text, to_gold_spans(gold_list), str(it.get("doc_id", "")),
                   str(it.get("qid", it.get("id", ""))))
        elif isinstance(it, Sequence) and not isinstance(it, str) and len(it) in (2, 3):
            gold = it[1]
            gold_list = [gold] if isinstance(gold, (str, GoldSpan)) else list(gold)
            q = QA(str(it[0]), to_gold_spans(gold_list), str(it[2]) if len(it) == 3 else "")
        else:
            raise TypeError(f"cannot read qa item {i}: {it!r}")
        if not q.gold:
            raise ValueError(f"qa item {i} has no gold evidence")
        q.qid = q.qid or str(i)
        out.append(q)
    if not out:
        raise ValueError("diagnose needs at least one labelled question")
    return out


def _to_chunks(corpus: Any) -> list[Chunk]:
    if isinstance(corpus, str):
        corpus = {"doc": corpus}
    if isinstance(corpus, Mapping):
        from .chunking import chunk

        return [c for doc_id, md in corpus.items() for c in chunk(md, doc_id=str(doc_id))]
    from .integrations import to_chunks

    try:
        return to_chunks(corpus)
    except TypeError as e:
        raise TypeError(f"{e}; or pass {{doc_id: markdown}}") from None


def _label(embedder: EmbedderLike, default: str) -> str:
    return embedder if isinstance(embedder, str) else default


# --------------------------------------------------------------------------- report

@dataclass
class ArmSummary:
    name: str
    mean: float
    ci95: tuple[float, float]
    n_questions: int


@dataclass
class Finding:
    """One comparison A vs B with its verdict and a plain-language message."""

    check: str
    a: str
    b: str
    comparison: stats.Comparison
    p_holm: float
    verdict: Literal["better", "worse", "equivalent", "identical", "inconclusive"]
    message: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "comparison"}
        d["comparison"] = self.comparison.to_dict()
        return d


@dataclass
class Report:
    """What ``diagnose`` found. ``print(report.to_markdown())`` for a readable summary."""

    arms: dict[str, ArmSummary]
    findings: list[Finding]
    per_question: dict[str, dict[str, list[float]]]  # arm -> cluster -> values (question order)
    question_ids: dict[str, list[str]]               # cluster -> qids
    settings: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def verdicts(self) -> list[str]:
        """One line per finding."""
        return [f.message for f in self.findings]

    def to_dict(self) -> dict[str, Any]:
        return {"arms": {k: dict(v.__dict__) for k, v in self.arms.items()},
                "findings": [f.to_dict() for f in self.findings],
                "per_question": self.per_question, "question_ids": self.question_ids,
                "settings": self.settings, "warnings": self.warnings}

    def to_markdown(self) -> str:
        s = self.settings
        metric = s["metric_label"]
        lines = ["# Retrieval diagnosis", "",
                 f"{s['n_questions']} questions in {s['n_clusters']} clusters ({s['cluster_by']}); "
                 f"metric: {metric}; scope: {s['scope']}; {s['n_boot']:,} bootstrap resamples; "
                 f"Holm-adjusted over {len(self.findings)} comparison(s).", ""]
        if self.warnings:
            lines += [f"> **Warning:** {w}" for w in self.warnings] + [""]
        if self.findings:
            lines += ["## Verdicts", ""] + [f"- {f.message}" for f in self.findings] + [""]
        lines += ["## Retrievers", "", f"| retriever | {metric} | 95% CI |", "|---|---|---|"]
        for a in sorted(self.arms.values(), key=lambda a: -a.mean):
            lines.append(f"| {a.name} | {_pt(a.mean)} | [{_pt(a.ci95[0])}, {_pt(a.ci95[1])}] |")
        if self.findings:
            lines += ["", "## Comparisons (A - B, points)", "",
                      "| check | A | B | diff | 95% CI | MDE | p | p (Holm) | verdict |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for f in self.findings:
                c = f.comparison
                lines.append(f"| {f.check} | {f.a} | {f.b} | {_spt(c.diff)} | "
                             f"[{_spt(c.ci95_bca[0])}, {_spt(c.ci95_bca[1])}] | {_mde(c)} | "
                             f"{c.p_perm:.3g} | {f.p_holm:.3g} | {f.verdict} |")
            ctx = list(dict.fromkeys(f.context for f in self.findings if f.context))
            lines += ["", "## Context from the v4 study", ""] + [f"- {c}" for c in ctx]
            lines.append(f"- {_SCOPE}")
        return "\n".join(lines) + "\n"


def _pt(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def _spt(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:+.1f}"


def _mde(c: stats.Comparison) -> str:
    if c.mde != c.mde:
        return "n/a (too few clusters)"
    if c.mde == 0:
        return "n/a (no variation)"
    return _pt(c.mde)


# --------------------------------------------------------------------------- running arms

Ranked = list[Chunk]


def _resolve(item: Any, lookup: Mapping[str, Chunk]) -> Chunk | None:
    """A retrieved item as a Chunk: Hit, Chunk, a chunk id (str), or any object
    ``integrations.to_chunks`` understands (LangChain Document, LlamaIndex node, Chonkie
    chunk, dict). A coerced item whose id is in the corpus resolves to the corpus chunk."""
    if isinstance(item, Hit):
        return item.chunk
    if isinstance(item, Chunk):
        return item
    if isinstance(item, str):
        return lookup.get(item)
    from .integrations import to_chunks

    try:
        c = to_chunks([item])[0]
    except TypeError:
        raise TypeError(f"a retriever returned {type(item).__name__}; expected Hit, Chunk, a "
                        "chunk id, or a LangChain/LlamaIndex/Chonkie chunk") from None
    return lookup.get(c.chunk_id, c)


def _call(r: Retriever | SearchFn, query: str, k: int) -> Sequence[RetrievedItem]:
    return r.search(query, k) if isinstance(r, Retriever) else r(query, k)


def _rerank(reranker: Any, query: str, ranked: Ranked, depth: int) -> Ranked:
    from .retrieval._base import index_text
    from .retrieval.rerank import rerank_order

    scores = reranker.scores(query, [index_text(c) for c in ranked[:depth]])
    order = rerank_order(list(range(len(ranked))), scores, depth)
    return [ranked[i] for i in order]


class _Runner:
    """Builds the built-in retrievers lazily per group (document or corpus) and runs arms."""

    def __init__(self, groups: dict[str, list[int]], chunks_by_group: dict[str, list[Chunk]],
                 qa: list[QA], k: int, bm25_method: str, embedder: EmbedderLike,
                 upgrade: EmbedderLike | None, reranker: Any, rerank_depth: int,
                 fusion: Any) -> None:
        self.groups, self.chunks_by_group, self.qa, self.k = groups, chunks_by_group, qa, k
        self.bm25_method, self.rerank_depth, self.fusion = bm25_method, rerank_depth, fusion
        self.embedders: dict[str, EmbedderLike] = {"dense": embedder}
        if upgrade is not None:
            self.embedders["dense-upgrade"] = upgrade
        self._embed_fns: dict[str, Any] = {}
        self.reranker = reranker
        self._built: dict[tuple[str, str], Any] = {}

    def _embed_fn(self, name: str) -> tuple[str, Any]:
        emb = self.embedders[name]
        if not isinstance(emb, str):
            return "custom", emb
        if name not in self._embed_fns:
            from .retrieval.dense import load_embedder

            self._embed_fns[name] = load_embedder(emb)
        return emb, self._embed_fns[name]

    def retriever(self, name: str, group: str, chunks: list[Chunk] | None = None) -> Any:
        from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

        key = (name, group if chunks is None else f"{group}#enriched")
        if key in self._built:
            return self._built[key]
        cs = self.chunks_by_group[group] if chunks is None else chunks
        if name == "bm25":
            r: Any = BM25Retriever(cs, method=self.bm25_method)
        elif name in self.embedders:
            model, fn = self._embed_fn(name)
            r = DenseRetriever(cs, model, embed_fn=fn)
        elif name == "hybrid":
            r = HybridRetriever(cs, bm25=self.retriever("bm25", group, chunks),
                                dense=self.retriever("dense", group, chunks), fusion=self.fusion)
        else:
            raise KeyError(name)
        self._built[key] = r
        return r

    def run_builtin(self, name: str, enriched: dict[str, list[Chunk]] | None = None
                    ) -> list[Ranked]:
        out: list[Ranked] = [[] for _ in self.qa]
        for group, qidx in self.groups.items():
            cs = enriched.get(group, []) if enriched is not None else None
            if (cs if cs is not None else self.chunks_by_group[group]) == []:
                continue
            r = self.retriever(name, group, cs)
            hits = r.search_many([self.qa[i].question for i in qidx], k=self.k)
            for i, h in zip(qidx, hits):
                out[i] = [x.chunk for x in h]
        return out

    def run_byo(self, r: Retriever | SearchFn, lookup: Mapping[str, Chunk],
                scope_docs: bool) -> list[Ranked]:
        out: list[Ranked] = []
        for q in self.qa:
            items = _call(r, q.question, self.k)
            ranked = [c for c in (_resolve(x, lookup) for x in items) if c is not None]
            if scope_docs and q.doc_id:  # within-document search: drop other documents' chunks
                ranked = [c for c in ranked if not c.doc_id or c.doc_id == q.doc_id]
            out.append(ranked)
        return out

    def rerank_arm(self, base: list[Ranked]) -> list[Ranked]:
        return [_rerank(self.reranker, q.question, ranked, self.rerank_depth) if ranked else []
                for q, ranked in zip(self.qa, base)]


# --------------------------------------------------------------------------- verdicts

def _verdict(c: stats.Comparison, p_holm: float, alpha: float, margin: float | None,
             identical: bool
             ) -> Literal["better", "worse", "equivalent", "identical", "inconclusive"]:
    if identical:
        return "identical"
    if c.excludes_zero_95 and p_holm < alpha:
        return "better" if c.diff > 0 else "worse"
    if margin and c.label in ("equivalent", "different_but_equivalent"):
        return "equivalent"
    return "inconclusive"


def _message(check: str, a: str, b: str, c: stats.Comparison, verdict: str, n_q: int,
             n_clusters: int, margin: float | None) -> str:
    est = f"{_spt(c.diff)} points [{_spt(c.ci95_bca[0])}, {_spt(c.ci95_bca[1])}]"
    mde = f"MDE {_mde(c)}" + (" points" if c.mde == c.mde and c.mde > 0 else "")
    if verdict == "identical":
        return (f"**No difference: identical on every question** ({a} vs {b}, n={n_q}). "
                "See the warnings: the budget may let every retriever find the evidence.")
    if verdict == "inconclusive":
        if c.mde != c.mde:
            reach = "Too few clusters to estimate the detectable difference"
        elif c.mde == 0:
            reach = "The difference does not vary across questions (MDE n/a)"
        else:
            reach = (f"Differences smaller than about {_pt(c.mde)} points cannot be detected "
                     "with these questions")
        return (f"**Inconclusive at n={n_q}** ({n_clusters} clusters): {a} vs {b}: {est}. "
                f"{reach}; label more questions to decide.")
    if verdict == "equivalent":
        return (f"**Equivalent within +/-{_pt(margin or 0)} points**: {a} vs {b}: {est} "
                f"(90% CI inside the margin; {mde}).")
    up = verdict == "better"
    advice = {
        ("fusion", False): f"Your fusion hurts: {a} scores below {b} alone. Tune the fusion "
                           "(RRF weights or k, or Convex) or drop the weaker half.",
        ("fusion", True): f"Fusion helps: {a} beats {b} alone.",
        ("embedder", True): f"A stronger embedder helps: {a} beats {b}. Upgrade the "
                            "retriever before paying for LLM indexing.",
        ("embedder", False): f"The upgrade embedder does not help here: {a} scores below {b}.",
        ("rerank", True): f"Reranking helps: {a} beats {b}.",
        ("rerank", False): f"Reranking hurts here: {a} scores below {b}.",
        ("enrich", True): f"Enrichment helps this retriever: {a} beats {b}.",
        ("enrich", False): f"Enrichment hurts this retriever: {a} scores below {b}.",
        ("custom", True): f"{a} beats {b}.",
        ("custom", False): f"{a} scores below {b}.",
    }[(check, up)]
    return f"**{advice}** {est}; {mde}."


# --------------------------------------------------------------------------- diagnose

def diagnose(corpus: Sequence[Chunk] | Sequence[Mapping[str, Any]] | Mapping[str, str] | str,
             qa: Iterable[Any], *,
             retrievers: Mapping[str, Retriever | SearchFn] | None = None,
             check: Sequence[str] | None = None,
             compare: Sequence[tuple[str, str]] = (),
             embedder: EmbedderLike = "qwen3-embedding-0.6b",
             upgrade_embedder: EmbedderLike | None = "qwen3-embedding-0.6b",
             reranker: Any = "qwen3-reranker-0.6b", rerank_depth: int = 30,
             bm25: str = "lucene", fusion: Any = None,
             enriched: Sequence[Chunk] | None = None, llm: Any = None,
             scope: Literal["auto", "document", "corpus"] = "auto",
             metric: Literal["recall", "precision"] = "recall",
             budget_tokens: int = 512, tokenizer: Tokenizer | str | None = None,
             margin: float | None = None, n_boot: int = 2000, seed: int = 0,
             alpha: float = 0.05, search_k: int = 50) -> Report:
    """Diagnose a retrieval stack on your documents and labelled questions.

    Args:
        corpus: chunks (``Chunk``, LangChain ``Document``, LlamaIndex node, Chonkie chunk, dict
            or string; see ``mdkeychunker.integrations.to_chunks``), ``{doc_id: markdown}``
            (chunked with ``mk.chunk``), or one Markdown string.
        qa: labelled questions: dicts ``{"question", "gold": [span text | GoldSpan |
            (doc_id, start, end)], "doc_id"?}``, ``(question, gold[, doc_id])`` tuples, or ``QA``.
        retrievers: your own retrievers, ``{name: Retriever or search(query, k) function}``,
            returning ``Hit``/``Chunk``/chunk ids or framework objects (LangChain documents,
            LlamaIndex nodes, ...; matched to the corpus by id). One becomes the baseline.
        check: any of ``"fusion"``, ``"embedder"``, ``"rerank"``, ``"enrich"``. Default:
            the first three, plus ``"enrich"`` when ``enriched`` or ``llm`` is given.
        compare: extra ``(a, b)`` pairs of arm names to compare.
        embedder: the built-in dense half: a preset / model id, or an ``embed_fn``.
        upgrade_embedder: the stronger embedder for ``"embedder"`` (skipped if the same as
            ``embedder`` and there is no retriever of your own).
        reranker: a reranker preset / model id or a ``Reranker`` (``"rerank"``).
        rerank_depth: results reordered by the reranker (study: 30).
        bm25: BM25 method (``"lucene"`` default; ``"study"`` for the paper's BM25).
        fusion: hybrid fusion (default ``RRF()``, the study's untuned k=60).
        enriched: the same documents' chunks after LLM enrichment (``"enrich"``).
        llm: alternatively, an LLM for ``mk.enrich`` to enrich the chunks here.
        scope: ``"document"`` searches each question within its ``doc_id`` (like Qasper);
            ``"corpus"`` searches everything; ``"auto"`` = document when every question has
            a ``doc_id``. Your retrievers' results from other documents are dropped in
            document scope.
        metric: ``"recall"`` (``evidence_recall``) or ``"precision"`` (``evidence_precision``).
        budget_tokens, tokenizer: the evidence budget (see ``evidence_recall``).
        margin: optional equivalence margin (in recall units, e.g. 0.05) for an
            "equivalent" verdict. None by default: the study's margins (0.05 Qasper, 0.04
            FreshStack) were set for its sample sizes and are not your thresholds.
        n_boot, seed, alpha: bootstrap resamples, seed and significance level.
        search_k: results requested per query (must cover the budget and the rerank depth).

    Returns:
        A ``Report``. A finding is "better"/"worse" only when its 95% BCa CI excludes zero
        and its Holm-adjusted sign-flip p is below ``alpha``; otherwise "inconclusive at
        n=..." (or "equivalent" when a margin is given and its 90% CI lies inside it). Every
        finding carries the minimum detectable difference.
    """
    if check is None:
        check = ("fusion", "embedder", "rerank") + (
            ("enrich",) if enriched is not None or llm is not None else ())
    unknown = set(check) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown check(s) {sorted(unknown)}; choose from {CHECKS}")
    qs = _to_qa(qa)
    chunks = _to_chunks(corpus)
    byo = dict(retrievers or {})
    reserved = {"bm25", "dense", "hybrid", "dense-upgrade"}
    if reserved & set(byo) or any(n.endswith(("+rerank", "@enriched")) for n in byo):
        raise ValueError(f"retriever names must not be {sorted(reserved)} or end in "
                         "'+rerank'/'@enriched'")

    by_doc = all(q.doc_id for q in qs)
    if scope == "auto":
        scope = "document" if by_doc else "corpus"
    if scope == "document":
        if not by_doc:
            raise ValueError("scope='document' needs a doc_id on every question")
        docs = {c.doc_id for c in chunks}
        missing = sorted({q.doc_id for q in qs} - docs)
        if missing:
            raise ValueError(f"no chunks for doc_id(s) {missing[:5]}")
    warnings: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(qs):
        groups.setdefault(q.doc_id if scope == "document" else "", []).append(i)
    chunks_by_group = ({g: [c for c in chunks if c.doc_id == g] for g in groups}
                       if scope == "document" else {"": chunks})
    k = max(search_k, rerank_depth)

    if fusion is None:
        from .retrieval.fusion import RRF

        fusion = RRF()
    if isinstance(reranker, str) and "rerank" in check:
        from .retrieval.rerank import Reranker

        reranker = Reranker(reranker)
    if "rerank" in check and reranker is None:
        raise ValueError("check='rerank' needs a reranker")
    same_upgrade = (isinstance(upgrade_embedder, str) and upgrade_embedder == embedder) or (
        upgrade_embedder is embedder)
    runner = _Runner(groups, chunks_by_group, qs, k, bm25, embedder,
                     None if same_upgrade else upgrade_embedder, reranker, rerank_depth, fusion)

    lookup = {c.chunk_id: c for c in chunks if c.chunk_id}
    results: dict[str, list[Ranked]] = {}

    def arm(name: str) -> list[Ranked]:
        if name not in results:
            if name in byo:
                results[name] = runner.run_byo(byo[name], lookup, scope == "document")
            elif name.endswith("+rerank"):
                results[name] = runner.rerank_arm(arm(name[: -len("+rerank")]))
            elif name.endswith("@enriched"):
                results[name] = runner.run_builtin(name[: -len("@enriched")], enriched_groups)
            else:
                results[name] = runner.run_builtin(name)
        return results[name]

    baseline = next(iter(byo)) if byo else "hybrid"
    if len(byo) > 1:
        warnings.append(f"Several retrievers given; '{baseline}' (the first) is the baseline.")
    pairs: list[tuple[str, str, str]] = []
    if "fusion" in check:
        pairs += [("fusion", "hybrid", "dense"), ("fusion", "hybrid", "bm25")]
    if "embedder" in check:
        if same_upgrade and not byo:
            warnings.append("check 'embedder' skipped: upgrade_embedder equals embedder, so it "
                            "is the fusion check's hybrid-vs-dense comparison.")
        else:
            pairs.append(("embedder", "dense" if same_upgrade else "dense-upgrade", baseline))
    if "rerank" in check:
        pairs.append(("rerank", f"{baseline}+rerank", baseline))
    enriched_groups: dict[str, list[Chunk]] | None = None
    if "enrich" in check or any(x.endswith("@enriched") for pair in compare for x in pair):
        enriched_chunks = _enriched(chunks, enriched, llm)
        enriched_groups = ({g: [c for c in enriched_chunks if c.doc_id == g] for g in groups}
                           if scope == "document" else {"": enriched_chunks})
    if "enrich" in check:
        pairs += [("enrich", "bm25@enriched", "bm25"), ("enrich", "hybrid@enriched", "hybrid")]
    pairs += [("custom", a, b) for a, b in compare]
    if not check and not compare and not byo:
        raise ValueError("nothing to do: give checks, compare pairs or retrievers")

    # score every arm that appears in a comparison (plus the user's retrievers)
    needed = list(dict.fromkeys([x for _, a, b in pairs for x in (a, b)] + list(byo)))
    _preflight(pairs, byo, embedder, upgrade_embedder, bm25, reranker)
    clusters = list(groups) if scope == "document" else [q.qid for q in qs]
    cluster_of = [q.doc_id if scope == "document" else q.qid for q in qs]
    score = evidence_recall if metric == "recall" else evidence_precision
    per: dict[str, dict[str, list[float]]] = {}
    for name in needed:
        ranked = arm(name)
        vals: dict[str, list[float]] = {c: [] for c in clusters}
        for q, cl, r in zip(qs, cluster_of, ranked):
            vals[cl].append(score(r, q.gold, budget_tokens, tokenizer=tokenizer))
        per[name] = vals

    n_q, n_c = len(qs), len(clusters)
    warnings += _ceiling_warnings(qs, groups, chunks_by_group, per, budget_tokens, tokenizer,
                                  scope)
    if n_q < 20:
        warnings.append(f"Only {n_q} questions: expect most comparisons to be inconclusive.")
    if n_c < 10:
        warnings.append(f"Only {n_c} clusters: bootstrap CIs are unreliable below ~10.")
    arms = {}
    for name, vals in per.items():
        boot = stats.cluster_bootstrap(vals, n=n_boot, seed=seed, levels=(0.95,))
        arms[name] = ArmSummary(name, boot.mean, boot.percentile[0.95], n_q)
    rows = [(chk, a, b, stats.compare(per[a], per[b], margin=margin, n=n_boot, seed=seed))
            for chk, a, b in pairs]
    p_adj, _ = stats.holm([r[3].p_perm for r in rows], alpha=alpha) if rows else ([], [])
    findings = []
    for (chk, a, b, c), ph in zip(rows, p_adj):
        v = _verdict(c, ph, alpha, margin, identical=per[a] == per[b])
        findings.append(Finding(chk, a, b, c, ph, v, _message(chk, a, b, c, v, n_q, n_c, margin),
                                _CONTEXT.get(chk, "")))
    metric_label = (f"evidence {'recall' if metric == 'recall' else 'precision'} within "
                    f"{budget_tokens} tokens")
    settings = {"n_questions": n_q, "n_clusters": n_c, "scope": scope,
                "cluster_by": "document" if scope == "document" else "question",
                "metric": metric, "metric_label": metric_label, "budget_tokens": budget_tokens,
                "n_boot": n_boot, "seed": seed, "alpha": alpha, "margin": margin,
                "baseline": baseline, "bm25": bm25,
                "embedder": _label(embedder, "custom"),
                "upgrade_embedder": None if upgrade_embedder is None
                else _label(upgrade_embedder, "custom"),
                "rerank_depth": rerank_depth, "search_k": k}
    qids: dict[str, list[str]] = {c: [] for c in clusters}
    for q, cl in zip(qs, cluster_of):
        qids[cl].append(q.qid)
    return Report(arms=arms, findings=findings, per_question=per, question_ids=qids,
                  settings=settings, warnings=warnings)


def _enriched(chunks: list[Chunk], enriched: Sequence[Chunk] | None, llm: Any) -> list[Chunk]:
    if enriched is not None:
        out = list(enriched)
        if {c.doc_id for c in out} != {c.doc_id for c in chunks}:
            raise ValueError("enriched chunks must cover the same documents as the corpus")
        return out
    if llm is None:
        raise ValueError("check='enrich' needs enriched= chunks or an llm=")
    try:
        enrich = importlib.import_module("mdkeychunker.enrich").enrich
    except (ImportError, AttributeError) as e:  # pragma: no cover - Session A's module
        raise ImportError("check='enrich' with llm= needs mk.enrich; pass enriched= instead"
                          ) from e
    return list(enrich(chunks, llm))


def _builtin_base(arm: str) -> str:
    return arm.split("+rerank")[0].split("@enriched")[0]


def _preflight(pairs: list[tuple[str, str, str]], byo: Mapping[str, Any], embedder: Any,
               upgrade: Any, bm25: str, reranker: Any) -> None:
    """Fail early, naming the check, when a built-in retriever's optional dependency is missing."""
    from importlib.util import find_spec

    def missing(mod: str) -> bool:
        return find_spec(mod) is None

    for chk, a, b in pairs:
        for arm in (a, b):
            if arm in byo:
                continue
            base = _builtin_base(arm)
            need_dense = base in ("dense", "hybrid", "dense-upgrade")
            emb = upgrade if base == "dense-upgrade" else embedder
            if need_dense and (missing("numpy") or (isinstance(emb, str)
                                                    and missing("sentence_transformers"))):
                raise ImportError(f"check='{chk}' needs a dense embedder: "
                                  'pip install "mdkeychunker[dense]" or pass embedder=your_fn')
            if base in ("bm25", "hybrid") and bm25 != "study" and missing("bm25s"):
                raise ImportError(f"check='{chk}' needs BM25: pip install \"mdkeychunker[bm25]\" "
                                  "or pass bm25='study' (no dependency)")
            if arm.endswith("+rerank") and getattr(reranker, "score_fn", None) is None and (
                    missing("sentence_transformers")):
                raise ImportError(f"check='{chk}' needs a reranker: "
                                  'pip install "mdkeychunker[rerank]" or pass reranker=Reranker('
                                  "score_fn=...)")


def _ceiling_warnings(qs: list[QA], groups: dict[str, list[int]],
                      chunks_by_group: dict[str, list[Chunk]],
                      per: dict[str, dict[str, list[float]]], budget_tokens: int,
                      tokenizer: Any, scope: str) -> list[str]:
    """Warn when the budget lets every retriever read (nearly) the whole candidate pool."""
    from .evaluate import CharTokenizer, get_tokenizer

    tok = get_tokenizer(tokenizer)

    def count(text: str) -> int:
        return tok.count(text) if isinstance(tok, CharTokenizer) else len(tok.encode(text))

    covered = 0
    for g, qidx in groups.items():
        pool = sum(count(c.text) for c in chunks_by_group[g])
        if pool <= budget_tokens:
            covered += len(qidx)
    out = []
    where = "whole documents" if scope == "document" else "the whole corpus"
    if covered >= len(qs) / 2:
        out.append(f"Your budget_tokens ({budget_tokens}) covers {where} for {covered} of "
                   f"{len(qs)} questions, so every retriever finds the evidence; lower "
                   "budget_tokens" + (" or use scope='corpus'." if scope == "document" else "."))
    means = [math.fsum(v for vs in arm.values() for v in vs) / len(qs) for arm in per.values()]
    if per and min(means) >= 0.99 and not out:
        out.append("Every retriever scores at least 99%: the questions are too easy to tell "
                   "retrievers apart at this budget; lower budget_tokens or add harder questions.")
    return out
