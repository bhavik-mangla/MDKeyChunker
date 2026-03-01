#!/usr/bin/env python3
"""Verification script for MDKeyChunker v2 installation."""
import sys


def check_imports():
    print("Checking imports...")
    try:
        from mdkeychunker import Pipeline, Chunk, Config
        print("  ✓ Core imports (Pipeline, Chunk, Config)")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_config():
    print("\nChecking configuration...")
    try:
        from mdkeychunker import Config
        config = Config()
        assert config.min_chunk_size > 0
        assert config.max_chunk_size > config.min_chunk_size
        assert config.llm_provider in (
            "openai", "anthropic", "openai_compatible")
        print("  ✓ Config defaults valid")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False


def check_chunker():
    print("\nChecking chunker...")
    try:
        from mdkeychunker.chunker import MarkdownChunker
        from mdkeychunker import Config
        chunker = MarkdownChunker(
            Config(min_chunk_size=50, max_chunk_size=500))
        chunks = chunker.chunk(
            "# Test\n\nParagraph content here.\n\n## Sub\n\nMore text.")
        assert len(chunks) > 0
        print(f"  ✓ Chunker working ({len(chunks)} chunks)")
        return True
    except Exception as e:
        print(f"  ✗ Chunker failed: {e}")
        return False


def check_llm_client():
    print("\nChecking LLM client...")
    try:
        from mdkeychunker.llm_client import LLMClient
        # Test JSON parsing (no API call needed)
        result = LLMClient._parse_json('{"key": "test"}')
        assert result == {"key": "test"}
        print("  ✓ LLM client JSON parsing works")
        return True
    except Exception as e:
        print(f"  ✗ LLM client failed: {e}")
        return False


def check_dependencies():
    print("\nChecking dependencies...")
    ok = True
    for pkg in ["openai", "anthropic", "dotenv", "tiktoken"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} not installed")
            ok = False
    return ok


def check_cli():
    print("\nChecking CLI...")
    try:
        from mdkeychunker.cli import main
        print("  ✓ CLI importable")
        return True
    except Exception as e:
        print(f"  ✗ CLI failed: {e}")
        return False


def main():
    print("=" * 50)
    print("MDKeyChunker v2 — Installation Verification")
    print("=" * 50 + "\n")

    checks = [check_imports, check_config, check_chunker,
              check_llm_client, check_dependencies, check_cli]
    results = [check() for check in checks]

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"Result: {passed}/{len(results)} checks passed")
    if passed == len(results):
        print("\n✓ Installation complete!")
        print("  Next: cp .env.sample .env && edit .env with your API key")
        print("  Then: mdkeychunker demo.md")
    else:
        print("\n✗ Some checks failed. Run: pip install -e .")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
