"""CLI for MDKeyChunker."""
import argparse
import sys
from pathlib import Path
from .config import Config
from .pipeline import Pipeline


def main() -> int:
    p = argparse.ArgumentParser(
        description="MDKeyChunker — Markdown chunking for RAG")
    p.add_argument("input", help="Input Markdown file")
    p.add_argument("-o", "--output", help="Output JSONL file")
    p.add_argument("--env", help="Path to .env file")
    p.add_argument("--stats", action="store_true", help="Print statistics")
    p.add_argument("--no-merge", action="store_true",
                   help="Disable key-based chunk merging")
    p.add_argument("--summary", help="Output summary text file")
    p.add_argument("--provider", choices=["openai", "anthropic", "openai_compatible"],
                   help="LLM provider (overrides .env)")
    p.add_argument("--model", help="LLM model name (overrides .env)")
    p.add_argument(
        "--base-url", help="LLM base URL for Ollama/vLLM (overrides .env)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    config = Config.from_env(args.env)

    if args.no_merge:
        config.merge_by_keys = False
    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        config.llm_model = args.model
    if args.base_url:
        config.llm_base_url = args.base_url

    pipeline = Pipeline(config)
    chunks = pipeline.process_file(str(input_path))

    out = args.output or str(input_path.with_suffix(".jsonl"))
    pipeline.save_jsonl(chunks, out)
    print(f"✓ {len(chunks)} chunks → {out}")

    if args.summary:
        pipeline.save_summary(chunks, args.summary)
        print(f"✓ Summary → {args.summary}")

    if args.stats:
        print(f"  Keys: {len(set(c.key for c in chunks if c.key))}")
        print(f"  Entities: {sum(len(c.entities) for c in chunks)}")
        print(
            f"  Avg tokens: {sum(c.token_count for c in chunks) // max(len(chunks), 1)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
