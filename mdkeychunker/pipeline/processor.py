"""
Main document processing pipeline.

Orchestrates chunking, metadata extraction, and linking.
"""

import logging
from pathlib import Path
from typing import List, Optional
import json

from ..models.chunk import Chunk, ChunkMetadata
from ..chunkers.markdown_chunker import MarkdownChunker
from ..metadata.entity_extractor import EntityExtractor
from ..metadata.keyword_extractor import KeywordExtractor
from ..metadata.numeric_extractor import NumericExtractor
from ..metadata.text_cleaner import TextCleaner
from ..linking.key_linker import KeyLinker
from ..llm.llm_client import LLMClient
from ..utils.config import Config
from ..utils.logger import setup_logger


class DocumentProcessor:
    """
    Main pipeline for processing Markdown documents.
    
    Supports two modes:
    - MODE A: Non-LLM (fast, deterministic)
    - MODE B: LLM-assisted (high accuracy)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.logger = setup_logger(
            level=self.config.log_level,
            log_file=self.config.log_file
        )
        
        # Initialize components
        self.chunker = MarkdownChunker(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.keyword_extractor = KeywordExtractor(self.config)
        self.numeric_extractor = NumericExtractor(self.config)
        self.text_cleaner = TextCleaner(self.config)
        self.key_linker = KeyLinker(self.config)
        
        # Initialize LLM client if budget > 0
        self.llm_client: Optional[LLMClient] = None
        if self.config.llm_call_budget_per_doc > 0:
            self.llm_client = LLMClient(self.config)
        
        self.logger.info("DocumentProcessor initialized")
    
    def process_file(self, filepath: str) -> List[Chunk]:
        """
        Process a Markdown file.
        
        Args:
            filepath: Path to Markdown file
            
        Returns:
            List of processed chunks with metadata
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not path.suffix.lower() in ['.md', '.markdown']:
            raise ValueError(f"Not a Markdown file: {filepath}")
        
        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.process_text(content)
    
    def process_text(self, markdown_text: str) -> List[Chunk]:
        """
        Process Markdown text.
        
        Args:
            markdown_text: Raw Markdown content
            
        Returns:
            List of processed chunks with complete metadata
        """
        self.logger.info("Starting document processing")
        
        # Step 1: Chunk the document
        self.logger.info("Chunking document...")
        chunks = self.chunker.chunk(markdown_text)
        self.logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Extract metadata for each chunk
        self.logger.info("Extracting metadata...")
        llm_calls_remaining = self.config.llm_call_budget_per_doc
        
        for i, chunk in enumerate(chunks):
            # Clean text
            cleaned_text, char_map = self.text_cleaner.clean(chunk.text)
            
            # Extract entities
            entities = self.entity_extractor.extract_with_mapping(
                chunk.text, cleaned_text, char_map
            )
            
            # Extract keywords
            keywords = self.keyword_extractor.extract(cleaned_text)
            
            # Extract numeric values
            numeric_values = self.numeric_extractor.extract(cleaned_text)
            
            # Generate summary (LLM or fallback)
            summary = ""
            if self.llm_client and llm_calls_remaining > 0:
                summary = self.llm_client.generate_summary(cleaned_text, chunk.section_title)
                llm_calls_remaining -= 1
            else:
                summary = self._fallback_summary(cleaned_text)
            
            # Generate questions if enabled
            questions = []
            if self.config.enable_questions and self.llm_client and llm_calls_remaining > 0:
                questions = self.llm_client.generate_questions(cleaned_text, chunk.section_title)
                llm_calls_remaining -= 1
            
            # Create metadata
            prev_id = chunks[i-1].metadata.chunk_id if i > 0 and chunks[i-1].metadata else None
            
            chunk.create_metadata(
                position_index=i,
                cleaned_text=cleaned_text,
                entities=entities,
                keywords=keywords,
                numeric_values=numeric_values,
                summary=summary,
                previous_chunk_id=prev_id,
                questions=questions
            )
        
        # Update next_chunk_id references
        for i in range(len(chunks) - 1):
            if chunks[i].metadata and chunks[i+1].metadata:
                chunks[i].metadata.next_chunk_id = chunks[i+1].metadata.chunk_id
        
        # Step 3: Pass-keys-forward linking
        if self.config.pass_keys_forward:
            self.logger.info("Running pass-keys-forward linking...")
            chunks = self.key_linker.process_chunks(
                chunks,
                llm_client=self.llm_client,
                llm_budget=llm_calls_remaining
            )
        
        self.logger.info(f"Processing complete. {len(chunks)} chunks with metadata.")
        return chunks
    
    def save_chunks(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Save chunks to JSONL file.
        
        Args:
            chunks: List of chunks with metadata
            output_path: Path to output JSONL file
        """
        path = Path(output_path)
        
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                if chunk.metadata:
                    f.write(chunk.metadata.to_json() + '\n')
        
        self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def save_summary(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Save document summary to text file.
        
        Args:
            chunks: List of chunks with metadata
            output_path: Path to output text file
        """
        path = Path(output_path)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("DOCUMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                if chunk.metadata and chunk.metadata.summary:
                    f.write(f"[Chunk {i+1}] {chunk.metadata.section_title or 'Untitled'}\n")
                    f.write(f"{chunk.metadata.summary}\n\n")
        
        self.logger.info(f"Saved summary to {output_path}")
    
    def _fallback_summary(self, text: str, max_length: int = 200) -> str:
        """Generate fallback summary without LLM."""
        # Take first sentence or two
        sentences = text.split('.')
        summary = sentences[0].strip()
        
        if len(summary) < 50 and len(sentences) > 1:
            summary += '. ' + sentences[1].strip()
        
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
        
        return summary + '.' if not summary.endswith('.') else summary
    
    def get_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Get processing statistics.
        
        Args:
            chunks: Processed chunks
            
        Returns:
            Dictionary with statistics
        """
        total_entities = sum(len(c.metadata.entities) for c in chunks if c.metadata)
        total_keywords = sum(len(c.metadata.keywords) for c in chunks if c.metadata)
        total_numeric = sum(len(c.metadata.numeric_values) for c in chunks if c.metadata)
        total_keys = sum(len(c.metadata.related_keys) for c in chunks if c.metadata)
        
        content_types = set()
        for chunk in chunks:
            content_types.update(chunk.content_types)
        
        return {
            "total_chunks": len(chunks),
            "total_entities": total_entities,
            "total_keywords": total_keywords,
            "total_numeric_values": total_numeric,
            "total_related_keys": total_keys,
            "content_types": list(content_types),
            "avg_entities_per_chunk": total_entities / len(chunks) if chunks else 0,
            "avg_keywords_per_chunk": total_keywords / len(chunks) if chunks else 0,
        }
