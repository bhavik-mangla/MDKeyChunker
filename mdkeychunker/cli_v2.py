"""Command line: ``mdkeychunker chunk`` and ``mdkeychunker diagnose``.

    mdkeychunker chunk docs/guide.md -o chunks.jsonl         # header-led chunks, free prefix
    mdkeychunker chunk notes.txt --windows --size 256        # token windows
    mdkeychunker diagnose --docs docs/ --qa qa.jsonl         # is the retriever leaving recall?
    mdkeychunker FILE                                        # 0.3 alias of `chunk` (no LLM)
    mdkeychunker legacy FILE --provider openai ...           # the deprecated 0.3 LLM pipeline

``qa.jsonl`` has one question per line: ``{"question": ..., "gold": [span text, or
{"doc_id", "start", "end"}, ...], "doc_id": ...}``. With ``--docs DIR`` a document's
``doc_id`` is its path relative to DIR (``guide/install.md``); with ``--chunks`` it is
whatever the chunks carry. No LLM is called unless you ask for the legacy pipeline.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, TextIO

from .models import Chunk

SUBCOMMANDS = ("chunk", "diagnose", "legacy")
DOC_SUFFIXES = (".md", ".markdown", ".mdx", ".txt")
# flags that only the 0.3 LLM pipeline understands: their presence routes FILE to `legacy`
LEGACY_FLAGS = ("--env", "--stats", "--no-merge", "--summary", "--provider", "--model",
                "--base-url")
_CHUNK_FIELDS = {f.name for f in fields(Chunk)}


# --------------------------------------------------------------------------- helpers

def _fail(msg: str) -> int:
    print(f"mdkeychunker: error: {msg}", file=sys.stderr)
    return 2


def _load_object(spec: str) -> Any:
    """``package.module:attribute`` -> the attribute (a function, retriever, embedder...)."""
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"expected module:attribute, got {spec!r}")
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _maybe_object(value: str | None) -> Any:
    """A preset name stays a string; ``module:attr`` is imported."""
    if value and ":" in value and not value.startswith(("http:", "https:")):
        return _load_object(value)
    return value


def _doc_files(paths: Sequence[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in DOC_SUFFIXES)
            out += [(f.relative_to(p).as_posix(), f) for f in files]
        elif p.is_file():
            out.append((p.name, p))
        else:
            raise FileNotFoundError(f"{p} not found")
    if not out:
        raise FileNotFoundError(f"no {'/'.join(DOC_SUFFIXES)} files in {', '.join(paths)}")
    return out


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not valid JSON ({e.msg})") from None
    return rows


def _chunks_from_jsonl(path: str) -> list[Chunk]:
    out = []
    for row in _read_jsonl(path):
        if "text" not in row:
            raise ValueError(f"{path}: every chunk needs 'text'")
        out.append(Chunk(**{k: v for k, v in row.items() if k in _CHUNK_FIELDS}))
    return out


def _write_chunks(chunks: Sequence[Chunk], out: TextIO) -> None:
    for c in chunks:
        out.write(c.to_json() + "\n")


# --------------------------------------------------------------------------- chunk

def _chunk_parser(prog: str = "mdkeychunker chunk") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Chunk Markdown or text files (no LLM).")
    p.add_argument("files", nargs="+", help="input files or directories")
    p.add_argument("-o", "--output", help="output JSONL (default: stdout)")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="cap per chunk's embed_text (default 512; 0 = no cap)")
    p.add_argument("--no-prefix", action="store_true",
                   help="embed_text without the section-path prefix")
    p.add_argument("--windows", action="store_true",
                   help="fixed token windows instead of header-led chunks")
    p.add_argument("--size", type=int, default=256, help="window size in tokens (--windows)")
    p.add_argument("--overlap", type=int, default=32, help="window overlap (--windows)")
    p.add_argument("--tokenizer", help="tiktoken encoding, e.g. cl100k_base (default chars/4)")
    return p


def _chunk_tokenizer(name: str | None) -> str | None:
    """CLI takes a bare tiktoken encoding name; mk.chunk spells it ``tiktoken:<name>``."""
    if not name or name.startswith("tiktoken"):
        return name
    return f"tiktoken:{name}"


def _run_chunk(args: argparse.Namespace, default_out: Path | None = None) -> int:
    from .chunking import chunk, window

    try:
        files = _doc_files(args.files)
    except FileNotFoundError as e:
        return _fail(str(e))
    chunks: list[Chunk] = []
    for doc_id, path in files:
        text = path.read_text(encoding="utf-8")
        if args.windows:
            chunks += window(text, size=args.size, overlap=args.overlap, doc_id=doc_id,
                             tokenizer=_chunk_tokenizer(args.tokenizer))
        else:
            chunks += chunk(text, doc_id=doc_id, max_tokens=args.max_tokens or None,
                            prefix=not args.no_prefix,
                            tokenizer=_chunk_tokenizer(args.tokenizer))
    out = args.output or (str(default_out) if default_out else None)
    if out and out != "-":
        with open(out, "w", encoding="utf-8") as fh:
            _write_chunks(chunks, fh)
        print(f"{len(chunks)} chunks from {len(files)} file(s) -> {out}", file=sys.stderr)
    else:
        _write_chunks(chunks, sys.stdout)
    return 0


# --------------------------------------------------------------------------- diagnose

def _diagnose_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdkeychunker diagnose",
        description="Compare retrievers on your documents and labelled questions, with "
                    "paired bootstrap CIs and 'inconclusive' when the data cannot tell.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs", nargs="+", help="Markdown/text files or directories (chunked)")
    src.add_argument("--chunks", help="chunks JSONL, e.g. from `mdkeychunker chunk`")
    p.add_argument("--qa", required=True, help="labelled questions JSONL")
    p.add_argument("--checks", default=None,
                   help="comma list of fusion,embedder,rerank,enrich (default: fusion,embedder,"
                        "rerank, plus enrich with --enriched); 'none' for only --compare")
    p.add_argument("--retriever", action="append", default=[], metavar="NAME=MODULE:ATTR",
                   help="your retriever: a search(query, k) function or Retriever object")
    p.add_argument("--compare", action="append", default=[], metavar="A,B",
                   help="extra comparison of two arms by name")
    p.add_argument("--enriched", help="LLM-enriched chunks JSONL (same documents) for 'enrich'")
    p.add_argument("--embedder", default="qwen3-embedding-0.6b",
                   help="dense preset / model id, or MODULE:ATTR embed function")
    p.add_argument("--upgrade-embedder", default="qwen3-embedding-0.6b",
                   help="stronger embedder for the 'embedder' check")
    p.add_argument("--reranker", default="qwen3-reranker-0.6b",
                   help="reranker preset / model id, or MODULE:ATTR Reranker")
    p.add_argument("--bm25", default="lucene", help="BM25 method: lucene (default) or study")
    p.add_argument("--scope", choices=("auto", "document", "corpus"), default="auto")
    p.add_argument("--metric", choices=("recall", "precision"), default="recall")
    p.add_argument("--budget", type=int, default=512, help="evidence budget in tokens")
    p.add_argument("--tokenizer", help="tiktoken encoding, e.g. cl100k_base (default chars/4)")
    p.add_argument("--margin", type=float, help="equivalence margin, e.g. 0.05 (optional)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", dest="json_out", help="also write the full report as JSON")
    p.add_argument("-o", "--output", help="write the Markdown report here instead of stdout")
    return p


def _run_diagnose(args: argparse.Namespace) -> int:
    from .chunking import chunk
    from .diagnose import diagnose

    try:
        if args.chunks:
            corpus: list[Chunk] = _chunks_from_jsonl(args.chunks)
        else:
            corpus = [c for doc_id, path in _doc_files(args.docs)
                      for c in chunk(path.read_text(encoding="utf-8"), doc_id=doc_id)]
        qa = _read_jsonl(args.qa)
        retrievers = {}
        for spec in args.retriever:
            name, sep, target = spec.partition("=")
            if not sep:
                raise ValueError(f"--retriever expects NAME=MODULE:ATTR, got {spec!r}")
            retrievers[name] = _load_object(target)
        compare = []
        for spec in args.compare:
            a, sep, b = spec.partition(",")
            if not sep:
                raise ValueError(f"--compare expects A,B, got {spec!r}")
            compare.append((a.strip(), b.strip()))
        checks = None
        if args.checks is not None:
            checks = () if args.checks == "none" else tuple(
                c.strip() for c in args.checks.split(",") if c.strip())
        enriched = _chunks_from_jsonl(args.enriched) if args.enriched else None
        report = diagnose(corpus, qa, retrievers=retrievers or None, check=checks,
                          compare=compare, embedder=_maybe_object(args.embedder),
                          upgrade_embedder=_maybe_object(args.upgrade_embedder),
                          reranker=_maybe_object(args.reranker), bm25=args.bm25,
                          enriched=enriched, scope=args.scope, metric=args.metric,
                          budget_tokens=args.budget, tokenizer=args.tokenizer,
                          margin=args.margin, n_boot=args.n_boot, seed=args.seed)
    except (FileNotFoundError, ValueError, TypeError, ImportError, AttributeError) as e:
        return _fail(str(e))
    md = report.to_markdown()
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"report -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- legacy / alias

def _run_legacy(argv: list[str]) -> int:
    warnings.warn("`mdkeychunker legacy` (the 0.3 LLM pipeline) is deprecated; use "
                  "`mdkeychunker chunk` and mk.enrich()", DeprecationWarning, stacklevel=2)
    print("mdkeychunker: note: running the deprecated 0.3 LLM pipeline", file=sys.stderr)
    from . import cli

    saved = sys.argv
    sys.argv = ["mdkeychunker legacy", *argv]
    try:
        return int(cli.main())
    finally:
        sys.argv = saved


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print((__doc__ or "").strip())
        return 0
    cmd = args[0]
    if cmd == "chunk":
        return _run_chunk(_chunk_parser().parse_args(args[1:]))
    if cmd == "diagnose":
        return _run_diagnose(_diagnose_parser().parse_args(args[1:]))
    if cmd == "legacy":
        return _run_legacy(args[1:])
    if any(a.split("=")[0] in LEGACY_FLAGS for a in args):
        print("mdkeychunker: note: LLM pipeline flags given; this runs the deprecated 0.3 "
              "pipeline. Prefer `mdkeychunker chunk` (no LLM) and mk.enrich().",
              file=sys.stderr)
        return _run_legacy(args)
    # 0.3 alias: `mdkeychunker FILE [-o OUT]` -> `chunk`, writing FILE.jsonl by default as before
    print("mdkeychunker: note: `mdkeychunker FILE` now runs `mdkeychunker chunk FILE` "
          "(header-led chunks, no LLM call). For the 0.3 LLM pipeline use "
          "`mdkeychunker legacy FILE`.", file=sys.stderr)
    ns = _chunk_parser(prog="mdkeychunker").parse_args(args)
    default = Path(ns.files[0]).with_suffix(".jsonl") if len(ns.files) == 1 else None
    return _run_chunk(ns, default_out=default)


if __name__ == "__main__":
    sys.exit(main())
