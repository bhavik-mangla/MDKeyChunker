"""MDKeyChunker: free structure-first chunking, reference retrieval and a retrieval diagnosis.

>>> import mdkeychunker as mk
>>> mk.chunk("# Guide\\n\\nInstall it.")[0].embed_text
'Guide\\n\\n# Guide\\n\\nInstall it.'

The core install has no dependencies. Names that need an optional backend (LLM clients,
retrievers) are imported on first use, so ``import mdkeychunker`` stays light.
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

from .chunking import chunk, chunk_file, window
from .enrich import Anthropic, OpenAICompatible, aenrich, enrich, estimate_cost
from .models import Chunk, GoldSpan, Hit, Retriever

try:
    __version__ = version("mdkeychunker")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0+unknown"

# Public name -> submodule, imported on first attribute access
_LAZY = {
    # evaluation, retrieval, diagnosis (session B)
    "evidence_recall": "evaluate", "evidence_precision": "evaluate",
    "BM25Retriever": "retrieval", "DenseRetriever": "retrieval",
    "HybridRetriever": "retrieval", "RRF": "retrieval", "Convex": "retrieval",
    "diagnose": "diagnose", "Report": "diagnose",
    # legacy 0.3 API (deprecated)
    "Pipeline": "pipeline", "Config": "config",
}

# enrich is imported eagerly: the submodule shares the function's name, and a lazy import
# would leave mk.enrich bound to the module. It imports no optional dependency.
__all__ = ["chunk", "chunk_file", "window", "enrich", "aenrich", "estimate_cost",
           "OpenAICompatible", "Anthropic", "Chunk", "GoldSpan", "Hit", "Retriever"] + [
    name for name, mod in _LAZY.items() if find_spec(f"{__name__}.{mod}") is not None]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
