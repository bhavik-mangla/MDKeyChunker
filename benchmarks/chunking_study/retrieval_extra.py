"""Secondary retrieval arms of Amendment 1, section A (ANALYSIS_PLAN_v2_AMENDMENT_1.md).

1. Reranking: Qwen3-Reranker-0.6B (sentence-transformers CrossEncoder,
   tomaarsen/Qwen3-Reranker-0.6B-seq-cls) reorders the hybrid top-30. Fallback
   BAAI/bge-reranker-v2-m3. The rerank input is the rendered index text (what the
   retriever indexed). Scores are cached under data/v2/rerank_cache/, keyed by
   (model + template/max_length, query hash, text hash).
2. qwen3-embedding:0.6b through Ollama /api/embed (query instruction from the
   model card; documents get none), and hybrid RRF(BM25, qwen3-embedding), k=60.
3. Late chunking with jinaai/jina-embeddings-v3: token embeddings of the whole
   document (8,192-token windows; longer documents use overlapping macro-windows),
   mean-pooled over each Stage-1 structural chunk's character span. Plain
   jina-v3 embeddings of rendered chunk texts are the comparison.

Heavy dependencies (torch, transformers, sentence-transformers) are imported
lazily, so the pure functions here (span location, pooling, windowing, RRF,
rerank ordering, cache keys) are testable without them. Run with .venv-ml.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

import evaluate

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
RERANK_CACHE = V2 / "rerank_cache"
JINA_CACHE = V2 / "jina_emb"
LC_CACHE = V2 / "lc_cache"

RRF_K = 60
RERANK_TOP = 30

# --------------------------------------------------------------------------- RRF

def rrf(*rankings: Sequence[int], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion; with two inputs identical to evaluate_v2.rrf / evaluate.rank('hybrid')."""
    score: dict[int, float] = {}
    for r_ in rankings:
        for pos, idx in enumerate(r_):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


def dense_orders(D: np.ndarray, Q: np.ndarray) -> list[list[int]]:
    """Cosine ranking (rows are L2-normalised), stable ties, as evaluate.rank."""
    return [list(np.argsort(-(D @ q), kind="stable")) for q in Q]


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def device_default() -> str:
    forced = os.environ.get("EXTRA_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- qwen3-embedding (Ollama)

QWEN3_EMB_MODEL = "qwen3-embedding:0.6b"
# Qwen3-Embedding model card: f'Instruct: {task_description}\nQuery:{query}', documents
# get no instruction. Default retrieval task description from the card (not tuned).
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_QUERY_PREFIX = f"Instruct: {QWEN3_TASK}\nQuery:"


def register_qwen3(ollama_url: str | None = None) -> None:
    """Register 'qwen3' as an evaluate.EMBEDDERS entry in this process only (no file edit).

    evaluate.embed then handles batching, truncation and the on-disk cache
    (data/emb, key = sha1(model|prefix+text)), exactly as for nomic and mxbai."""
    evaluate.EMBEDDERS["qwen3"] = (QWEN3_EMB_MODEL, QWEN3_QUERY_PREFIX, "")
    # num_batch: an embedding input longer than one micro-batch (default 2,048 tokens) crashes the runner
    evaluate.EMBED_OPTIONS[QWEN3_EMB_MODEL] = {"num_ctx": QWEN3_NUM_CTX, "num_batch": QWEN3_NUM_CTX}
    if ollama_url:
        evaluate.OLLAMA = ollama_url


# Ollama loads qwen3-embedding with a 4,096-token context by default, and its runner crashes on longer
# inputs instead of honouring truncate=True. We load it with num_ctx 8,192 (embeddings of shorter
# texts are bit-identical) and cut the rare longer document to 3,000 Qwen3 tokens client-side (the runner still crashed on some texts of 3,800-3,950 tokens at num_ctx 8,192).
QWEN3_NUM_CTX = 8192
QWEN3_MAX_DOC_TOKENS = 3000
QWEN3_TOKENIZER = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"  # same Qwen3 vocabulary, already cached
_QTOK = None


def qwen3_truncate(text: str, max_tokens: int = QWEN3_MAX_DOC_TOKENS) -> str:
    global _QTOK
    if len(text) <= max_tokens:  # a token is at least one character
        return text
    if _QTOK is None:
        from transformers import AutoTokenizer
        _QTOK = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
    ids = _QTOK(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else _QTOK.decode(ids[:max_tokens])


def qwen3_orders(docs: list[str], queries: list[str]) -> list[list[int]]:
    if "qwen3" not in evaluate.EMBEDDERS:
        register_qwen3()
    # FreshStack queries are whole Stack Overflow posts, so they get the same cut as documents
    return evaluate.rank("qwen3", [qwen3_truncate(d) for d in docs], [qwen3_truncate(q) for q in queries])


# --------------------------------------------------------------------------- reranker

QWEN3_RERANKER = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
              'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
              '<|im_start|>user\n')
_QR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_VERSION = "v1"


def format_pair(model_name: str, query: str, doc: str) -> tuple[str, str]:
    """The Qwen3 reranker needs its chat template applied by hand (model card); bge takes raw text."""
    if "qwen3-reranker" in model_name.lower():
        return (f"{_QR_PREFIX}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n",
                f"<Document>: {doc}{_QR_SUFFIX}")
    return query, doc


class RerankCache:
    """Scores on disk: <root>/<model_key_hash>/<qh[:2]>/<qh>.json = {text_hash: score}.

    model_key includes the model id, template version and max_length, so a change
    of any of them can never reuse old scores."""

    def __init__(self, root: Path, model_key: str):
        self.root = Path(root) / sha1(model_key)[:16]
        self.model_key = model_key
        self._mem: dict[str, dict[str, float]] = {}

    @staticmethod
    def query_hash(query: str) -> str:
        return sha1("q|" + query)

    @staticmethod
    def text_hash(text: str) -> str:
        return sha1("t|" + text)

    def _path(self, qh: str) -> Path:
        return self.root / qh[:2] / f"{qh}.json"

    def load(self, query: str) -> dict[str, float]:
        qh = self.query_hash(query)
        if qh not in self._mem:
            p = self._path(qh)
            self._mem[qh] = json.loads(p.read_text()) if p.exists() else {}
        return self._mem[qh]

    def save(self, query: str, scores: dict[str, float]) -> None:
        qh = self.query_hash(query)
        cur = self.load(query)
        cur.update(scores)
        p = self._path(qh)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not (self.root / "model_key.txt").exists():
            (self.root / "model_key.txt").write_text(self.model_key)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(cur))
        tmp.replace(p)


ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


@dataclass
class Reranker:
    """Cross-encoder reranker with a persistent score cache.

    score_fn maps formatted (query, doc) pairs to scores; by default it is the
    sentence-transformers CrossEncoder, loaded lazily (tests inject a fake)."""
    model_name: str = QWEN3_RERANKER
    max_length: int = 2048
    batch_size: int = 16
    device: str | None = None
    cache_root: Path = RERANK_CACHE
    score_fn: ScoreFn | None = None
    pairs_scored: int = 0
    seconds: float = 0.0
    cache_hits: int = 0
    _ce: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cache = RerankCache(self.cache_root, self.model_key)

    @property
    def model_key(self) -> str:
        return f"{self.model_name}|tmpl={TEMPLATE_VERSION}|maxlen={self.max_length}"

    def _load(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder
        dev = self.device or device_default()
        self.device = dev
        kw = {"torch_dtype": torch.float16 if dev == "mps" else torch.float32}
        try:
            self._ce = CrossEncoder(self.model_name, device=dev, max_length=self.max_length, model_kwargs=kw)
        except Exception as e:  # pragma: no cover - exercised only on real hardware
            if dev != "cpu":
                print(f"WARN reranker on {dev} failed ({e}); falling back to CPU", flush=True)
                self.device = "cpu"
                self._ce = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length,
                                        model_kwargs={"torch_dtype": torch.float32})
            else:
                raise
        tok = self._ce.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.score_fn is not None:
            return list(map(float, self.score_fn(pairs)))
        if self._ce is None:
            self._load()
        # sort by length so batches pad little; predict keeps input order
        perm = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        out = self._ce.predict([pairs[i] for i in perm], batch_size=self.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
        res = [0.0] * len(pairs)
        for j, i in enumerate(perm):
            res[i] = float(np.asarray(out[j]).reshape(-1)[0])
        return res

    def scores(self, query: str, texts: list[str]) -> list[float]:
        cached = self.cache.load(query)
        hs = [RerankCache.text_hash(t) for t in texts]
        todo = sorted({h: t for h, t in zip(hs, texts) if h not in cached}.items())
        self.cache_hits += len(texts) - len(todo)
        if todo:
            t0 = time.time()
            vals = self._predict([format_pair(self.model_name, query, t) for _, t in todo])
            self.seconds += time.time() - t0
            self.pairs_scored += len(todo)
            self.cache.save(query, {h: v for (h, _), v in zip(todo, vals)})
            cached = self.cache.load(query)
        return [cached[h] for h in hs]

    def rerank(self, order: Sequence[int], query: str, docs: list[str], top: int = RERANK_TOP) -> list[int]:
        return rerank_order(order, self.scores(query, [docs[i] for i in order[:top]]), top)

    def stats(self) -> dict:
        return {"model": self.model_name, "model_key": self.model_key, "pairs_scored": self.pairs_scored,
                "cache_hits": self.cache_hits, "seconds": round(self.seconds, 2),
                "pairs_per_s": round(self.pairs_scored / self.seconds, 2) if self.seconds else None,
                "device": self.device}


def rerank_order(order: Sequence[int], head_scores: Sequence[float], top: int = RERANK_TOP) -> list[int]:
    """Reorder order[:top] by score (descending; ties keep the first-stage rank); the tail is unchanged."""
    head = list(order[:top])
    assert len(head_scores) == len(head)
    idx = sorted(range(len(head)), key=lambda j: (-head_scores[j], j))
    return [head[j] for j in idx] + list(order[top:])


# --------------------------------------------------------------------------- span location

_WS = re.compile(r"\s+")


def locate_spans(md: str, texts: Sequence[str]) -> tuple[list[tuple[int, int] | None], dict]:
    """Character spans of chunk texts in the document markdown.

    Search order per chunk: exact match from a running cursor (chunks are in
    document order, so duplicates resolve to the right occurrence), exact match
    anywhere, then a whitespace-insensitive match (stripped text, any run of
    whitespace matching any other). Returns None for chunks that cannot be found."""
    spans: list[tuple[int, int] | None] = []
    counts = {"exact_cursor": 0, "exact_anywhere": 0, "whitespace_normalised": 0, "not_found": 0, "empty": 0}
    cur = 0
    for t in texts:
        if not t.strip():
            spans.append(None)
            counts["empty"] += 1
            continue
        i = md.find(t, cur)
        kind = "exact_cursor"
        if i < 0:
            i = md.find(t)
            kind = "exact_anywhere"
        if i >= 0:
            spans.append((i, i + len(t)))
            counts[kind] += 1
            cur = i + len(t) if kind == "exact_cursor" else max(cur, i + len(t))
            continue
        pat = r"\s+".join(re.escape(w) for w in _WS.split(t.strip()))
        m = re.compile(pat).search(md, cur) or re.compile(pat).search(md)
        if m:
            spans.append((m.start(), m.end()))
            counts["whitespace_normalised"] += 1
            cur = max(cur, m.end())
        else:
            spans.append(None)
            counts["not_found"] += 1
    return spans, counts


def token_range(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> tuple[int, int] | None:
    """First and last (inclusive) token index whose character offsets overlap span [a, b)."""
    a, b = span
    hit = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
    return (hit[0], hit[-1]) if hit else None


# --------------------------------------------------------------------------- late chunking

class TokenEncoder(Protocol):
    """What late chunking needs from a long-context encoder (jina-v3 or a test fake)."""
    max_len: int

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...
    def wrap(self, content_ids: list[int], task: str) -> tuple[list[int], int]:
        """Full input ids for one window and the index of its first content token."""
    def token_embeddings(self, input_ids: list[int], task: str) -> np.ndarray: ...
    def content_capacity(self, task: str) -> int: ...


def macro_windows(n: int, capacity: int, overlap: int) -> list[tuple[int, int]]:
    """Token windows [s, e) over n content tokens; one window when it fits, else overlapping
    windows of `capacity` tokens with stride capacity - overlap, the last aligned to the end."""
    if n <= capacity:
        return [(0, n)]
    if overlap >= capacity:
        raise ValueError("overlap must be smaller than the window capacity")
    stride = capacity - overlap
    out, s = [], 0
    while s + capacity < n:
        out.append((s, s + capacity))
        s += stride
    out.append((n - capacity, n))
    return out


def best_window(tr: tuple[int, int], windows: list[tuple[int, int]]) -> int | None:
    """Window that fully contains token range tr with the most context (max of min(left, right))."""
    i, j = tr
    best, best_ctx = None, -1
    for w, (s, e) in enumerate(windows):
        if s <= i and j < e:
            ctx = min(i - s, e - 1 - j)
            if ctx > best_ctx:
                best, best_ctx = w, ctx
    return best


def token_best_window(t: int, windows: list[tuple[int, int]]) -> int:
    return max((w for w, (s, e) in enumerate(windows) if s <= t < e),
               key=lambda w: min(t - windows[w][0], windows[w][1] - 1 - t))


@dataclass
class LateChunkReport:
    n_chunks: int = 0
    n_windows: int = 0
    pooled: int = 0
    split_across_windows: int = 0
    fallback_plain: int = 0
    span_counts: dict = field(default_factory=dict)

    def add(self, o: "LateChunkReport") -> None:
        self.n_chunks += o.n_chunks
        self.n_windows += o.n_windows
        self.pooled += o.pooled
        self.split_across_windows += o.split_across_windows
        self.fallback_plain += o.fallback_plain
        for k, v in o.span_counts.items():
            self.span_counts[k] = self.span_counts.get(k, 0) + v

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def late_chunk(enc: TokenEncoder, md: str, chunk_texts: Sequence[str], task: str = "retrieval.passage",
               overlap: int = 2048, plain_fn: Callable[[list[str]], np.ndarray] | None = None
               ) -> tuple[np.ndarray, LateChunkReport]:
    """Late-chunked embeddings (L2-normalised) for the chunks of one document.

    Each chunk is mean-pooled over the token embeddings of the window that fully
    contains it with the most context; a chunk wider than every window takes each
    token from the window where that token has the most context (counted as
    split). Chunks whose span cannot be located (or that cover no token) fall
    back to plain_fn (a plain embedding of the chunk text) and are counted."""
    spans, counts = locate_spans(md, chunk_texts)
    rep = LateChunkReport(n_chunks=len(chunk_texts), span_counts=counts)
    ids, offs = enc.tokenize(md)
    windows = macro_windows(len(ids), enc.content_capacity(task), overlap)
    rep.n_windows = len(windows)
    ranges = [token_range(offs, sp) if sp else None for sp in spans]

    need: dict[int, list[int]] = {}
    plan: list = []
    for c, tr in enumerate(ranges):
        if tr is None:
            plan.append(None)
            continue
        w = best_window(tr, windows)
        if w is not None:
            plan.append([(t, w) for t in range(tr[0], tr[1] + 1)])
            need.setdefault(w, [])
        else:
            assign = [(t, token_best_window(t, windows)) for t in range(tr[0], tr[1] + 1)]
            plan.append(assign)
            rep.split_across_windows += 1
            for _, w2 in assign:
                need.setdefault(w2, [])
    win_emb: dict[int, tuple[np.ndarray, int, int]] = {}
    for w in sorted(need):
        s, e = windows[w]
        full, first = enc.wrap(ids[s:e], task)
        win_emb[w] = (enc.token_embeddings(full, task), first, s)

    out: list[np.ndarray | None] = []
    for assign in plan:
        if assign is None:
            out.append(None)
            continue
        vecs = []
        for t, w in assign:
            E, first, s = win_emb[w]
            vecs.append(E[first + (t - s)].astype(np.float32))
        out.append(_norm(np.mean(vecs, axis=0)))
        rep.pooled += 1
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        rep.fallback_plain = len(missing)
        if plain_fn is None:
            raise ValueError(f"{len(missing)} chunks could not be located and no plain_fn was given")
        P = plain_fn([chunk_texts[i] for i in missing])
        for i, v in zip(missing, P):
            out[i] = _norm(v)
    return np.vstack(out), rep


class JinaV3:
    """jinaai/jina-embeddings-v3 with task LoRA adapters (transformers, trust_remote_code).

    The task instruction (config.task_instructions) is prepended exactly as the
    model's own encode() does, for passages and queries alike; token embeddings
    come from model.forward with adapter_mask, mean pooling as in encode()."""
    name = "jinaai/jina-embeddings-v3"

    def __init__(self, device: str | None = None, max_len: int = 8192, batch_size: int = 8):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or device_default()
        self.max_len = max_len
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        # the remote config rejects a torch.dtype kwarg under transformers 5; cast after loading
        self.model = AutoModel.from_pretrained(self.name, trust_remote_code=True)
        self.model.to(device=self.device, dtype=dtype).eval()
        self.instructions = dict(self.model.config.task_instructions)
        self.adapt = dict(self.model._adaptation_map)
        self._prefix: dict[str, list[int]] = {}

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def prefix_ids(self, task: str) -> list[int]:
        if task not in self._prefix:
            self._prefix[task] = list(self.tok(self.instructions[task], add_special_tokens=False)["input_ids"])
        return self._prefix[task]

    def content_capacity(self, task: str) -> int:
        return self.max_len - 2 - len(self.prefix_ids(task))

    def wrap(self, content_ids: list[int], task: str) -> tuple[list[int], int]:
        pre = self.prefix_ids(task)
        return [self.tok.cls_token_id] + pre + list(content_ids) + [self.tok.sep_token_id], 1 + len(pre)

    def token_embeddings(self, input_ids: list[int], task: str) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            ids = t.tensor([input_ids], device=self.device)
            mask = t.ones_like(ids)
            am = t.full((1,), self.adapt[task], dtype=t.int32, device=self.device)
            out = self.model(input_ids=ids, attention_mask=mask, adapter_mask=am)[0]
            return out[0].float().cpu().numpy()

    def encode(self, texts: list[str], task: str) -> np.ndarray:
        """Plain embeddings (the model's own encode: instruction, mean pooling, L2 norm, 8,192 tokens)."""
        try:
            return np.asarray(self.model.encode(texts, task=task, batch_size=self.batch_size,
                                                max_length=self.max_len), dtype=np.float32)
        except RuntimeError as e:  # a batch of several 8k-token texts can exceed MPS memory
            if "out of memory" not in str(e):
                raise
            if self.device == "mps":
                self.torch.mps.empty_cache()
            return np.vstack([np.asarray(self.model.encode([t], task=task, batch_size=1,
                                                           max_length=self.max_len), dtype=np.float32)
                              for t in texts])


def cached_encode(texts: list[str], task: str, encode_fn: Callable[[list[str], str], np.ndarray],
                  model_name: str = JinaV3.name, root: Path = JINA_CACHE) -> np.ndarray:
    """Per-text embedding cache keyed by sha1(model|task|text)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        p = root / f"{sha1(f'{model_name}|{task}|{t}')}.npy"
        if p.exists():
            out[i] = np.load(p)
        else:
            todo.append((i, p))
    if todo:
        # longest first keeps padding low inside the encoder's batches
        todo.sort(key=lambda x: -len(texts[x[0]]))
        V = encode_fn([texts[i] for i, _ in todo], task)
        for (i, p), v in zip(todo, V):
            v = _norm(v)
            np.save(p, v)
            out[i] = v
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def lc_cache_path(doc_id: str, md: str, chunk_texts: Sequence[str], overlap: int, max_len: int,
                  root: Path = LC_CACHE) -> Path:
    key = sha1(json.dumps([JinaV3.name, md, list(chunk_texts), overlap, max_len]))[:16]
    return Path(root) / f"{doc_id}.{key}.npz"


def late_chunk_cached(enc: TokenEncoder, doc_id: str, md: str, chunk_texts: Sequence[str],
                      overlap: int = 2048, plain_fn: Callable | None = None,
                      root: Path = LC_CACHE) -> tuple[np.ndarray, LateChunkReport]:
    p = lc_cache_path(doc_id, md, chunk_texts, overlap, enc.max_len, root)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        rep = LateChunkReport(**json.loads(str(z["report"])))
        return z["emb"], rep
    E, rep = late_chunk(enc, md, chunk_texts, overlap=overlap, plain_fn=plain_fn)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, emb=E, report=np.array(json.dumps(rep.as_dict())))
    return E, rep
