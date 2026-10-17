"""End-to-end smoke test with two tiny public models (a few MB): the library's own
load_embedder / DenseRetriever / Reranker on whatever versions are installed. CI's
lowest-direct job runs it to prove the [dense] / [rerank] floors actually work.

    uv run --resolution lowest-direct --extra retrieval python tests/smoke_tiny_models.py
"""
import importlib.metadata as md
import math

from mdkeychunker.models import Chunk
from mdkeychunker.retrieval import DenseRetriever, Reranker
from mdkeychunker.retrieval._base import model_dtype_kwargs
from mdkeychunker.retrieval.dense import load_embedder

EMBEDDER = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
CROSS_ENCODER = "cross-encoder-testing/reranker-bert-tiny-gooaq-bce"


def main() -> None:
    print({p: md.version(p) for p in ("torch", "transformers", "sentence-transformers", "numpy")})
    print("dtype kwargs:", list(model_dtype_kwargs("float32")))
    embed = load_embedder(EMBEDDER, device="cpu", batch_size=2)
    vecs = embed(["hello world", "goodbye moon"])
    assert len(vecs) == 2 and all(abs(math.sqrt(sum(float(x) ** 2 for x in v)) - 1) < 1e-3
                                  for v in vecs), vecs
    r = DenseRetriever([Chunk(text="a cat", chunk_id="c"), Chunk(text="a car", chunk_id="d")],
                       EMBEDDER, embed_fn=embed)
    assert len(r.search("cat", k=2)) == 2
    scores = Reranker(CROSS_ENCODER, device="cpu").scores("cat", ["a cat", "a car"])
    assert len(scores) == 2 and all(math.isfinite(s) for s in scores), scores
    print("tiny-model smoke OK:", [round(s, 4) for s in scores])


if __name__ == "__main__":
    main()
