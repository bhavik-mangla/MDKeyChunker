"""Importing the evaluation/retrieval modules must not pull in heavy optional dependencies."""
import subprocess
import sys

HEAVY = ("numpy", "torch", "sentence_transformers", "transformers", "bm25s", "Stemmer",
         "tiktoken", "openai", "dotenv")


def _imported_after(stmt: str) -> list[str]:
    code = (f"import sys\n{stmt}\n"
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return [m for m in out.stdout.strip().split(",") if m]


# What `import mdkeychunker` itself loads is Session A's (A2 removes dotenv/openai/tiktoken);
# these tests check that the retrieval modules add nothing on top of it.
def _added_by(stmt: str) -> set[str]:
    return set(_imported_after(stmt)) - set(_imported_after("import mdkeychunker"))


def test_retrieval_package_import_is_light():
    assert _added_by("import mdkeychunker.retrieval") == set()


def test_fusion_needs_nothing():
    assert _added_by("from mdkeychunker.retrieval import RRF, Convex") == set()


def test_study_bm25_runs_without_numpy_or_bm25s():
    code = ("import sys\n"
            "sys.modules['numpy'] = None\nsys.modules['bm25s'] = None\n"
            "from mdkeychunker.models import Chunk\n"
            "from mdkeychunker.retrieval import BM25Retriever\n"
            "cs = [Chunk(text='install on linux'), Chunk(text='pricing plans')]\n"
            "print(BM25Retriever(cs, method='study').search('linux', k=1)[0].chunk.text)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert out.stdout.strip() == "install on linux"
