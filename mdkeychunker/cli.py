"""Command-line interface for MDKeyChunker."""

import argparse
import sys
from pathlib import Path
import json

from .pipeline.processor import DocumentProcessor
from .utils.config import Config


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MDKeyChunker - Markdown document chunking with metadata enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (uses .env file)
  mdkeychunker input.md

  # Specify output location
  mdkeychunker input.md -o output.jsonl

  # Use custom config file
  mdkeychunker input.md --env custom.env

  # Disable LLM (MODE A - fast, deterministic)
  mdkeychunker input.md --no-llm

  # Show statistics
  mdkeychunker input.md --stats
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input Markdown file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: input_name.jsonl)"
    )
    
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Output summary file path (default: input_name_summary.txt)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env configuration file"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (MODE A: fast, deterministic)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print processing statistics"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Determine output paths
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.jsonl')
    
    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = input_path.parent / f"{input_path.stem}_summary.txt"
    
    try:
        # Load configuration
        config = Config.from_env(args.env)
        
        # Override with CLI args
        if args.no_llm:
            config.llm_call_budget_per_doc = 0
        
        if args.log_level:
            config.log_level = args.log_level
        
        # Initialize processor
        print(f"Processing: {input_path}")
        processor = DocumentProcessor(config)
        
        # Process document
        chunks = processor.process_file(str(input_path))
        
        # Save outputs
        processor.save_chunks(chunks, str(output_path))
        print(f"✓ Chunks saved to: {output_path}")
        
        processor.save_summary(chunks, str(summary_path))
        print(f"✓ Summary saved to: {summary_path}")
        
        # Print statistics if requested
        if args.stats:
            stats = processor.get_statistics(chunks)
            print("\nProcessing Statistics:")
            print("=" * 50)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, list):
                    print(f"  {key}: {', '.join(value)}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n✓ Processing complete! {len(chunks)} chunks generated.")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
