"""Core data models for chunks and metadata."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import hashlib
import json


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    span_start: int
    span_end: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NumericValue:
    """Represents an extracted numeric value."""
    value: float
    unit: Optional[str]
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ChunkMetadata:
    """Complete metadata for a chunk."""
    chunk_id: str
    text: str
    cleaned_text: str
    section_title: str
    entities: List[Entity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    numeric_values: List[NumericValue] = field(default_factory=list)
    summary: str = ""
    position_index: int = 0
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    content_types: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    related_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "section_title": self.section_title,
            "entities": [e.to_dict() for e in self.entities],
            "keywords": self.keywords,
            "numeric_values": [nv.to_dict() for nv in self.numeric_values],
            "summary": self.summary,
            "position_index": self.position_index,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "content_types": self.content_types,
            "questions": self.questions,
            "related_keys": self.related_keys,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Chunk:
    """Represents a document chunk with its metadata."""
    text: str
    section_title: str = ""
    start_line: int = 0
    end_line: int = 0
    content_types: List[str] = field(default_factory=list)
    metadata: Optional[ChunkMetadata] = None

    @staticmethod
    def generate_chunk_id(text: str, section_title: str, position_index: int) -> str:
        """Generate a stable chunk ID based on content and position."""
        # Create a stable identifier from normalized content + position
        normalized = f"{section_title}:{position_index}:{text[:100]}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def create_metadata(
        self,
        position_index: int,
        cleaned_text: str,
        entities: List[Entity],
        keywords: List[str],
        numeric_values: List[NumericValue],
        summary: str = "",
        previous_chunk_id: Optional[str] = None,
        next_chunk_id: Optional[str] = None,
        questions: Optional[List[str]] = None,
        related_keys: Optional[List[str]] = None,
    ) -> ChunkMetadata:
        """Create and attach metadata to this chunk."""
        chunk_id = self.generate_chunk_id(self.text, self.section_title, position_index)
        
        self.metadata = ChunkMetadata(
            chunk_id=chunk_id,
            text=self.text,
            cleaned_text=cleaned_text,
            section_title=self.section_title,
            entities=entities,
            keywords=keywords,
            numeric_values=numeric_values,
            summary=summary,
            position_index=position_index,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            content_types=self.content_types,
            questions=questions or [],
            related_keys=related_keys or [],
        )
        return self.metadata
