"""Example usage of MDKeyChunker."""

from mdkeychunker import DocumentProcessor
from mdkeychunker.utils.config import Config
import json


def example_mode_a():
    """Example: Fast, deterministic processing (no LLM)."""
    print("=" * 60)
    print("MODE A: Non-LLM Processing")
    print("=" * 60)
    
    # Configure for MODE A (no LLM)
    config = Config(
        llm_call_budget_per_doc=0,  # Disable LLM
        entity_extraction_mode="none",  # Skip spaCy for speed (optional)
        pass_keys_forward=True,
        log_level="INFO"
    )
    
    processor = DocumentProcessor(config)
    
    # Process demo file
    print("\nProcessing demo.md...")
    chunks = processor.process_file("demo.md")
    
    # Save outputs
    processor.save_chunks(chunks, "chunks_mode_a.jsonl")
    processor.save_summary(chunks, "summary_mode_a.txt")
    
    # Show statistics
    stats = processor.get_statistics(chunks)
    print(f"\n✓ Processed {stats['total_chunks']} chunks")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Keywords: {stats['total_keywords']}")
    print(f"  Numeric values: {stats['total_numeric_values']}")
    print(f"  Related keys: {stats['total_related_keys']}")
    
    # Show first chunk as example
    if chunks:
        print("\nFirst chunk preview:")
        print(json.dumps(chunks[0].metadata.to_dict(), indent=2)[:500] + "...")


def example_mode_b():
    """Example: LLM-assisted processing (requires API key)."""
    print("\n" + "=" * 60)
    print("MODE B: LLM-Assisted Processing")
    print("=" * 60)
    
    # Configure for MODE B (with LLM)
    config = Config.from_env()  # Load from .env file
    
    # Override for example
    config.llm_call_budget_per_doc = 20
    config.enable_questions = True
    
    print(f"\nLLM Provider: {config.llm_provider}")
    print(f"Budget: {config.llm_call_budget_per_doc} calls")
    
    # Check if API key is set
    if not config.llm_api_key or config.llm_api_key == "your_api_key_here":
        print("\n⚠️  WARNING: LLM API key not configured!")
        print("   Set LLM_API_KEY in .env file to use MODE B")
        print("   Continuing with MODE A instead...\n")
        example_mode_a()
        return
    
    processor = DocumentProcessor(config)
    
    # Process demo file
    print("\nProcessing demo.md with LLM enhancement...")
    chunks = processor.process_file("demo.md")
    
    # Save outputs
    processor.save_chunks(chunks, "chunks_mode_b.jsonl")
    processor.save_summary(chunks, "summary_mode_b.txt")
    
    # Show statistics
    stats = processor.get_statistics(chunks)
    print(f"\n✓ Processed {stats['total_chunks']} chunks")
    
    # Show chunk with questions
    chunks_with_questions = [c for c in chunks if c.metadata.questions]
    if chunks_with_questions:
        print("\nExample chunk with generated questions:")
        chunk = chunks_with_questions[0]
        print(f"Section: {chunk.metadata.section_title}")
        print(f"Questions: {chunk.metadata.questions}")


def example_custom_config():
    """Example: Custom configuration for specific use case."""
    print("\n" + "=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)
    
    # Configure for technical documentation
    config = Config(
        llm_call_budget_per_doc=0,
        entity_extraction_mode="spacy",
        strict_entity_mode=True,
        min_chunk_size=50,
        soft_max_chunk_size=800,
        hard_max_chunk_size=1500,
        entity_weight=0.5,  # Emphasize entities in technical docs
        keyword_weight=0.3,
        proximity_weight=0.15,
        ref_weight=0.05,
        threshold_accept=0.65,  # More conservative linking
        log_level="DEBUG"
    )
    
    processor = DocumentProcessor(config)
    
    print("\nConfiguration:")
    print(f"  Chunk size: {config.soft_max_chunk_size} (soft) / {config.hard_max_chunk_size} (hard)")
    print(f"  Entity weight: {config.entity_weight}")
    print(f"  Accept threshold: {config.threshold_accept}")
    
    # Process
    chunks = processor.process_file("demo.md")
    stats = processor.get_statistics(chunks)
    
    print(f"\n✓ Generated {stats['total_chunks']} chunks")


def example_programmatic_access():
    """Example: Programmatic access to chunk data."""
    print("\n" + "=" * 60)
    print("Programmatic Access Example")
    print("=" * 60)
    
    config = Config(llm_call_budget_per_doc=0)
    processor = DocumentProcessor(config)
    
    chunks = processor.process_file("demo.md")
    
    # Analyze chunks
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Find chunks with code
    code_chunks = [c for c in chunks if 'code' in c.content_types]
    print(f"Code blocks: {len(code_chunks)}")
    
    # Find chunks with tables
    table_chunks = [c for c in chunks if 'table' in c.content_types]
    print(f"Tables: {len(table_chunks)}")
    
    # Find most entity-rich chunks
    chunks_by_entities = sorted(
        chunks,
        key=lambda c: len(c.metadata.entities) if c.metadata else 0,
        reverse=True
    )
    
    print("\nTop 3 entity-rich chunks:")
    for i, chunk in enumerate(chunks_by_entities[:3]):
        if chunk.metadata:
            print(f"  {i+1}. {chunk.metadata.section_title}")
            print(f"     Entities: {[e.name for e in chunk.metadata.entities[:5]]}")
    
    # Analyze keyword distribution
    all_keywords = set()
    for chunk in chunks:
        if chunk.metadata:
            all_keywords.update(chunk.metadata.keywords)
    
    print(f"\nUnique keywords across document: {len(all_keywords)}")
    print(f"Sample: {list(all_keywords)[:10]}")


if __name__ == "__main__":
    # Run examples
    example_mode_a()
    
    # Uncomment to run other examples:
    # example_mode_b()  # Requires LLM API key
    # example_custom_config()
    # example_programmatic_access()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
