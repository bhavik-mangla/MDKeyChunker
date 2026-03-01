"""Example usage of MDKeyChunker v2."""

import json
from mdkeychunker import Pipeline, Config


def example_basic():
    """Basic usage: chunk a Markdown file and save as JSONL."""
    print("=" * 60)
    print("Basic Example: Process demo.md")
    print("=" * 60)

    # Config loads from .env by default; override as needed
    config = Config.from_env()
    pipeline = Pipeline(config)

    chunks = pipeline.process_file("demo.md")
    pipeline.save_jsonl(chunks, "chunks.jsonl")
    pipeline.save_summary(chunks, "summary.txt")

    print(f"\n✓ {len(chunks)} chunks → chunks.jsonl")
    print(f"  Unique keys : {len(set(c.key for c in chunks if c.key))}")
    print(f"  Total entities: {sum(len(c.entities) for c in chunks)}")
    print(
        f"  Avg tokens  : {sum(c.token_count for c in chunks) // max(len(chunks), 1)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(json.dumps(chunks[0].to_dict(), indent=2,
              ensure_ascii=False)[:600] + "...")


def example_ollama():
    """Use a local Ollama model instead of OpenAI."""
    print("\n" + "=" * 60)
    print("Ollama (Local LLM) Example")
    print("=" * 60)

    config = Config(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        llm_model="llama3",
    )
    pipeline = Pipeline(config)
    chunks = pipeline.process_file("demo.md")
    print(f"✓ {len(chunks)} chunks processed with Ollama")


def example_no_merge():
    """Disable key-based merging for raw chunks."""
    print("\n" + "=" * 60)
    print("No-Merge Example")
    print("=" * 60)

    config = Config.from_env()
    config.merge_by_keys = False
    pipeline = Pipeline(config)
    chunks = pipeline.process_file("demo.md")
    print(f"✓ {len(chunks)} unmerged chunks")


def example_programmatic():
    """Inspect chunks programmatically."""
    print("\n" + "=" * 60)
    print("Programmatic Access Example")
    print("=" * 60)

    config = Config.from_env()
    pipeline = Pipeline(config)
    chunks = pipeline.process_text(
        "# Intro\n\nMachine learning is a subset of AI.\n\n"
        "## Details\n\nNeural networks learn representations.\n\n"
        "## Conclusion\n\nML powers modern applications.\n"
    )

    print(f"Total chunks: {len(chunks)}")
    code_chunks = [c for c in chunks if "code" in c.content_types]
    print(f"Code chunks : {len(code_chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}: {chunk.title or chunk.section_title}")
        print(f"  Key     : {chunk.key}")
        print(f"  Keywords: {chunk.keywords[:3]}")
        print(f"  Entities: {[e['name'] for e in chunk.entities[:3]]}")


if __name__ == "__main__":
    example_basic()
    # example_ollama()   # Requires Ollama running locally
    # example_no_merge()
    # example_programmatic()
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

