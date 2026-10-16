"""BM25 over chunks: bm25s (``pip install "mdkeychunker[bm25]"``) or the study's exact Okapi."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from ..models import Chunk
from ._base import BaseRetriever, argsort_desc, index_text, require
from .fusion import Ranking

_WORD = re.compile(r"[a-z0-9]+")


def study_tokenize(text: str) -> list[str]:
    """The study's BM25 tokenisation: lower-cased ASCII alphanumeric runs, no stemming."""
    return _WORD.findall(text.lower())


class _Okapi:
    """rank-bm25 ``BM25Okapi`` (the study's BM25), ported to give bit-identical scores.

    idf = ln(N - n + 0.5) - ln(n + 0.5); negative idfs are replaced by
    ``epsilon * mean(idf)``. Repeated query terms count once per occurrence.
    """

    def __init__(self, corpus: list[list[str]], k1: float, b: float, epsilon: float) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / self.n_docs
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf: dict[str, float] = {}
        negative = []
        idf_sum = 0.0  # a running total, not sum(): Python >= 3.12 sum() compensates rounding
        for term, plist in self.postings.items():
            n = len(plist)
            idf = math.log(self.n_docs - n + 0.5) - math.log(n + 0.5)
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative.append(term)
        eps = epsilon * (idf_sum / len(self.idf))
        for term in negative:
            self.idf[term] = eps

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n_docs
        k1, b = self.k1, self.b
        for q in query:
            idf = self.idf.get(q) or 0.0
            for i, tf in self.postings.get(q, ()):
                norm = tf + k1 * (1 - b + b * self.doc_len[i] / self.avgdl)
                out[i] += idf * (tf * (k1 + 1) / norm)
        return out


class BM25Retriever(BaseRetriever):
    """BM25 over each chunk's ``embed_text`` (or ``text``).

    Args:
        chunks: the chunks to index.
        k1, b: BM25 parameters (study: 1.5, 0.75).
        method: ``"lucene"`` (default; the bm25s default and the Lucene/Elasticsearch/
            OpenSearch formula, so ``diagnose`` verdicts transfer to those stacks), any
            other bm25s method (``"robertson"``, ``"atire"``, ``"bm25+"``, ``"bm25l"``), or
            ``"study"``: the study's rank-bm25 ``BM25Okapi`` (idf floor ``epsilon``),
            reproduced with bit-identical scores and needing no dependency at all.
            **The default is not the paper's setting.** Measured on Qasper (free prefix,
            all 951 questions, recall within 512 cl100k tokens): ``"study"`` 0.2989 (the
            paper's value), ``"lucene"`` 0.3179, ``"atire"`` + ``idf_method="robertson"``
            0.2944. Use ``"study"`` to reproduce paper numbers.
        idf_method: bm25s idf variant (``None`` = the method's own).
        epsilon: idf floor factor for ``method="study"`` (rank-bm25 default 0.25).
        tokenize: text -> tokens; defaults to ``study_tokenize``.
        stemmer: a PyStemmer language such as ``"english"`` (needs PyStemmer), or ``None``.

    Example:
        >>> from mdkeychunker.models import Chunk
        >>> cs = [Chunk(text="install on linux"), Chunk(text="pricing plans")]
        >>> BM25Retriever(cs, method="study").search("linux install", k=1)[0].chunk.text
        'install on linux'
    """

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75,
                 method: str = "lucene", idf_method: str | None = None, epsilon: float = 0.25,
                 tokenize: Callable[[str], list[str]] | None = None,
                 stemmer: str | None = None) -> None:
        super().__init__(chunks)
        self.method = method
        self._tokenize = tokenize or study_tokenize
        self._stem: Callable[[str], str] | None = None
        if stemmer:
            stem_mod = require("Stemmer", "bm25")
            self._stem = stem_mod.Stemmer(stemmer).stemWord
        # an empty chunk still needs a token so every chunk keeps its index (as in the study)
        corpus = [self._tokens(index_text(c)) or ["_"] for c in self.chunks]
        self._okapi: _Okapi | None = None
        self._bm25: Any = None
        if method == "study":
            if corpus:
                self._okapi = _Okapi(corpus, k1, b, epsilon)
        else:
            bm25s = require("bm25s", "bm25")
            self._bm25 = bm25s.BM25(k1=k1, b=b, method=method, idf_method=idf_method)
            if corpus:
                self._bm25.index(corpus, show_progress=False)

    def _tokens(self, text: str) -> list[str]:
        toks = self._tokenize(text)
        return [self._stem(t) for t in toks] if self._stem else toks

    def scores(self, query: str) -> list[float]:
        """BM25 score of every chunk for ``query`` (0 for chunks sharing no term)."""
        if not self.chunks:
            return []
        toks = self._tokens(query)
        if self._okapi is not None:
            return self._okapi.scores(toks)
        vocab = self._bm25.vocab_dict
        q = [t for t in toks if t in vocab]
        if not q:  # bm25s raises on an empty query
            return [0.0] * len(self.chunks)
        return [float(s) for s in self._bm25.get_scores(q)]

    def rank_many(self, queries: Sequence[str]) -> list[Ranking]:
        return [argsort_desc(self.scores(q)) if self.chunks else [] for q in queries]
