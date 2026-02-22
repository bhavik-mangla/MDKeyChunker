"""
MDKeyChunker - Markdown-only document chunking with metadata enrichment.

A production-ready package for chunking Markdown documents while preserving
structure and extracting rich metadata for RAG pipelines.
"""

__version__ = "0.1.0"

from .pipeline.processor import DocumentProcessor
from .models.chunk import Chunk, ChunkMetadata

__all__ = ["DocumentProcessor", "Chunk", "ChunkMetadata"]
