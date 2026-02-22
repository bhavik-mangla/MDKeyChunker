#!/usr/bin/env python3
"""
Verification script for MDKeyChunker installation.
Run this after installation to verify everything works.
"""

import sys
from pathlib import Path


def check_import():
    """Test basic imports."""
    print("Checking imports...")
    try:
        from mdkeychunker import DocumentProcessor
        from mdkeychunker.utils.config import Config
        from mdkeychunker.models.chunk import Chunk, Entity, NumericValue
        print("  ✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_config():
    """Test configuration loading."""
    print("\nChecking configuration...")
    try:
        from mdkeychunker.utils.config import Config
        
        # Test default config
        config = Config()
        assert config.min_chunk_size > 0
        assert config.soft_max_chunk_size > config.min_chunk_size
        print("  ✓ Configuration system working")
        return True
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        return False


def check_chunker():
    """Test basic chunking functionality."""
    print("\nChecking chunker...")
    try:
        from mdkeychunker.chunkers.markdown_chunker import MarkdownChunker
        from mdkeychunker.utils.config import Config
        
        config = Config(
            min_chunk_size=50,
            soft_max_chunk_size=500,
            hard_max_chunk_size=1000
        )
        chunker = MarkdownChunker(config)
        
        # Test simple markdown
        markdown = """# Test Header

This is a test paragraph.

## Subsection

Another paragraph here.
"""
        chunks = chunker.chunk(markdown)
        assert len(chunks) > 0
        print(f"  ✓ Chunker working ({len(chunks)} chunks created)")
        return True
    except Exception as e:
        print(f"  ✗ Chunker failed: {e}")
        return False


def check_metadata_extraction():
    """Test metadata extraction."""
    print("\nChecking metadata extraction...")
    try:
        from mdkeychunker.metadata.keyword_extractor import KeywordExtractor
        from mdkeychunker.metadata.numeric_extractor import NumericExtractor
        from mdkeychunker.metadata.text_cleaner import TextCleaner
        from mdkeychunker.utils.config import Config
        
        config = Config()
        
        # Test keyword extraction
        kw_extractor = KeywordExtractor(config)
        keywords = kw_extractor.extract("Machine learning and deep learning are important.")
        
        # Test numeric extraction
        num_extractor = NumericExtractor(config)
        numbers = num_extractor.extract("The model achieved 95.5% accuracy.")
        
        # Test text cleaning
        cleaner = TextCleaner(config)
        cleaned, _ = cleaner.clean("Text  with   extra    spaces")
        
        print("  ✓ Metadata extraction working")
        return True
    except Exception as e:
        print(f"  ✗ Metadata extraction failed: {e}")
        return False


def check_llm_client():
    """Test LLM client initialization."""
    print("\nChecking LLM client...")
    try:
        from mdkeychunker.llm.llm_client import LLMClient
        from mdkeychunker.llm.prompt_templates import PromptTemplates
        from mdkeychunker.utils.config import Config
        
        config = Config(llm_call_budget_per_doc=0)  # No actual calls
        client = LLMClient(config)
        
        # Test prompt generation
        templates = PromptTemplates()
        prompt = templates.summary_generation("Test text", "Test Section")
        assert len(prompt) > 0
        
        print("  ✓ LLM client initialized")
        return True
    except Exception as e:
        print(f"  ✗ LLM client failed: {e}")
        return False


def check_pipeline():
    """Test full pipeline."""
    print("\nChecking pipeline...")
    try:
        from mdkeychunker.pipeline.processor import DocumentProcessor
        from mdkeychunker.utils.config import Config
        
        config = Config(
            llm_call_budget_per_doc=0,  # No LLM calls
            entity_extraction_mode="none",  # Skip spaCy for speed
            log_level="ERROR"  # Quiet
        )
        
        processor = DocumentProcessor(config)
        
        # Test with simple markdown
        markdown = """# Test Document

This is a test with some content about machine learning.

## Results

The model achieved 95% accuracy.
"""
        
        chunks = processor.process_text(markdown)
        assert len(chunks) > 0
        
        # Check metadata
        assert chunks[0].metadata is not None
        assert chunks[0].metadata.chunk_id is not None
        
        # Get statistics
        stats = processor.get_statistics(chunks)
        assert stats['total_chunks'] == len(chunks)
        
        print(f"  ✓ Pipeline working ({len(chunks)} chunks generated)")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cli():
    """Check CLI availability."""
    print("\nChecking CLI...")
    try:
        from mdkeychunker.cli import main
        print("  ✓ CLI available")
        return True
    except Exception as e:
        print(f"  ✗ CLI failed: {e}")
        return False


def check_spacy():
    """Check spaCy (optional)."""
    print("\nChecking spaCy (optional)...")
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("  ✓ spaCy model available")
            return True
        except OSError:
            print("  ⚠ spaCy installed but model not downloaded")
            print("    Run: python -m spacy download en_core_web_sm")
            return True
    except ImportError:
        print("  ⚠ spaCy not installed (optional)")
        return True


def check_demo_file():
    """Check if demo file exists."""
    print("\nChecking demo file...")
    demo_path = Path("demo.md")
    if demo_path.exists():
        print("  ✓ demo.md found")
        return True
    else:
        print("  ⚠ demo.md not found (should be in package root)")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("MDKeyChunker Installation Verification")
    print("=" * 60)
    print()
    
    checks = [
        check_import,
        check_config,
        check_chunker,
        check_metadata_extraction,
        check_llm_client,
        check_pipeline,
        check_cli,
        check_spacy,
        check_demo_file,
    ]
    
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✓ All checks passed! Installation is complete.")
        print("\nNext steps:")
        print("  1. Copy .env.sample to .env and configure")
        print("  2. Run: python examples.py")
        print("  3. Process a document: mdkeychunker demo.md")
        return 0
    else:
        print("\n⚠ Some checks failed. Please review errors above.")
        print("\nFor help, see README.md or open an issue.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
