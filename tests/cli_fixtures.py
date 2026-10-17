"""Objects the CLI tests load by MODULE:ATTR (no model downloads)."""
from mdkeychunker.retrieval import Reranker


def flat_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def never(query, k):
    return []


null_reranker = Reranker(score_fn=lambda pairs: [0.0 for _ in pairs])
