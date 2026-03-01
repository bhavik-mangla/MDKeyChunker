"""Data models for MDKeyChunker."""
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class Chunk:
    text: str
    section_title: str = ""
    content_types: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    # Metadata (populated by enricher)
    chunk_id: str = ""
    title: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)  # [{name, type}]
    questions: list[str] = field(default_factory=list)
    key: str = ""
    related_keys: list[str] = field(default_factory=list)
    position_index: int = 0
    previous_chunk_id: str = ""
    next_chunk_id: str = ""
    token_count: int = 0

    def generate_id(self) -> str:
        h = hashlib.sha256(
            f"{self.section_title}:{self.key}:{self.position_index}:{self.text[:100]}".encode(
            )
        ).hexdigest()[:16]
        self.chunk_id = h
        return h

    def to_dict(self) -> dict:
        """Return all fields. Empty lists and zero values are preserved."""
        return dict(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, key={self.key!r}, "
            f"title={self.title!r}, tokens={self.token_count})"
        )
