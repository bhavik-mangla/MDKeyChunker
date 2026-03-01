"""
Markdown structural chunker.

Splits Markdown documents following structure (headers, lists, tables, code blocks)
without breaking semantic units.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .models import Chunk
from .config import Config


@dataclass
class _Block:
    """Internal structural block in Markdown."""
    type: str  # header, paragraph, list, code, table, blockquote, yaml_front_matter
    content: str
    start_line: int
    end_line: int
    level: int = 0  # For headers
    section_title: str = ""


class MarkdownChunker:
    """
    Chunks Markdown documents following structural boundaries.

    Never splits:
    - Code blocks
    - Tables
    - YAML front matter
    - List items mid-item
    - Blockquotes mid-paragraph
    """

    def __init__(self, config: Config):
        self.config = config
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self._header_stack: List[Tuple[int, str]] = []

    def chunk(self, markdown_text: str) -> List[Chunk]:
        """Chunk a Markdown document into structural units."""
        blocks = self._parse_blocks(markdown_text)
        chunks = self._group_blocks(blocks)
        return chunks

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_blocks(self, text: str) -> List[_Block]:
        """Parse Markdown into structural blocks."""
        blocks: List[_Block] = []
        lines = text.split("\n")
        i = 0
        current_section = ""
        self._header_stack = []

        while i < len(lines):
            line = lines[i]

            # YAML front matter
            if i == 0 and line.lstrip("\ufeff").strip() == "---":
                block, consumed = self._parse_yaml_front_matter(lines, i)
                if block:
                    blocks.append(block)
                    i += consumed
                    continue

            # Fenced code blocks
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                block, consumed = self._parse_code_block(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += consumed
                continue

            # Headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                self._update_header_stack(level, title)
                current_section = self._get_current_section()
                blocks.append(_Block(
                    type="header",
                    content=line,
                    start_line=i,
                    end_line=i,
                    level=level,
                    section_title=current_section,
                ))
                i += 1
                continue

            # Tables
            if self._is_table_line(line, lines, i):
                block, consumed = self._parse_table(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += consumed
                continue

            # Lists
            if self._is_list_line(line):
                block, consumed = self._parse_list(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += consumed
                continue

            # Blockquotes
            if line.strip().startswith(">"):
                block, consumed = self._parse_blockquote(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += consumed
                continue

            # Horizontal rules
            if re.match(r"^(\*\*\*+|---+|___+)\s*$", line):
                blocks.append(_Block(
                    type="horizontal_rule",
                    content=line,
                    start_line=i,
                    end_line=i,
                    section_title=current_section,
                ))
                i += 1
                continue

            # Paragraphs
            if line.strip():
                block, consumed = self._parse_paragraph(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += consumed
                continue

            i += 1  # empty line

        return blocks

    # ------------------------------------------------------------------ #
    # Block parsers                                                        #
    # ------------------------------------------------------------------ #

    def _parse_yaml_front_matter(
        self, lines: List[str], start: int
    ) -> Tuple[Optional[_Block], int]:
        end = start + 1
        while end < len(lines) and lines[end].strip() != "---":
            end += 1
        if end >= len(lines):
            return None, 0
        content = "\n".join(lines[start: end + 1])
        return _Block(
            type="yaml_front_matter",
            content=content,
            start_line=start,
            end_line=end,
        ), end - start + 1

    def _parse_code_block(self, lines: List[str], start: int) -> Tuple[_Block, int]:
        fence = lines[start].strip()[:3]
        end = start + 1
        while end < len(lines):
            if lines[end].strip().startswith(fence):
                end += 1
                break
            end += 1
        content = "\n".join(lines[start:end])
        return _Block(
            type="code", content=content, start_line=start, end_line=end - 1
        ), end - start

    def _is_table_line(self, line: str, lines: list[str], idx: int) -> bool:
        """Check if this line starts a table (header row followed by separator)."""
        if "|" not in line:
            return False
        # Must have a separator line immediately after — only detect from the header row
        if idx + 1 < len(lines):
            return bool(re.match(r"^\s*\|?[\s:-]+\|[\s|:-]+\|?\s*$", lines[idx + 1]))
        return False

    def _parse_table(self, lines: List[str], start: int) -> Tuple[_Block, int]:
        end = start
        while end < len(lines) and ("|" in lines[end] and lines[end].strip()):
            end += 1
        content = "\n".join(lines[start:end])
        return _Block(
            type="table", content=content, start_line=start, end_line=end - 1
        ), end - start

    def _is_list_line(self, line: str) -> bool:
        stripped = line.strip()
        return bool(
            re.match(r"^[\*\-\+]\s+",
                     stripped) or re.match(r"^\d+\.\s+", stripped)
        )

    def _parse_list(self, lines: List[str], start: int) -> Tuple[_Block, int]:
        end = start
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        end += 1
        while end < len(lines):
            line = lines[end]
            if not line.strip():
                peek = end + 1
                while peek < len(lines) and not lines[peek].strip():
                    peek += 1
                if peek < len(lines) and (
                    self._is_list_line(lines[peek])
                    or len(lines[peek]) - len(lines[peek].lstrip()) > base_indent
                ):
                    end += 1
                    continue
                break
            indent = len(line) - len(line.lstrip())
            if self._is_list_line(line) or indent > base_indent:
                end += 1
            else:
                break
        content = "\n".join(lines[start:end])
        return _Block(
            type="list", content=content, start_line=start, end_line=end - 1
        ), end - start

    def _parse_blockquote(self, lines: List[str], start: int) -> Tuple[_Block, int]:
        end = start
        while end < len(lines) and lines[end].strip().startswith(">"):
            end += 1
        content = "\n".join(lines[start:end])
        return _Block(
            type="blockquote", content=content, start_line=start, end_line=end - 1
        ), end - start

    def _parse_paragraph(self, lines: List[str], start: int) -> Tuple[_Block, int]:
        end = start
        while end < len(lines):
            line = lines[end]
            if not line.strip():
                break
            if end > start and (
                line.strip().startswith("#")
                or line.strip().startswith("```")
                or line.strip().startswith("~~~")
                or self._is_list_line(line)
                or line.strip().startswith(">")
                or "|" in line
            ):
                break
            end += 1
        content = "\n".join(lines[start:end])
        return _Block(
            type="paragraph", content=content, start_line=start, end_line=end - 1
        ), max(end - start, 1)

    # ------------------------------------------------------------------ #
    # Header stack                                                         #
    # ------------------------------------------------------------------ #

    def _update_header_stack(self, level: int, title: str) -> None:
        self._header_stack = [(l, t)
                              for l, t in self._header_stack if l < level]
        self._header_stack.append((level, title))

    def _get_current_section(self) -> str:
        return " > ".join(t for _, t in self._header_stack)

    # ------------------------------------------------------------------ #
    # Grouping                                                             #
    # ------------------------------------------------------------------ #

    def _group_blocks(self, blocks: List[_Block]) -> List[Chunk]:
        """Group blocks into chunks respecting size constraints."""
        chunks: List[Chunk] = []
        current: List[_Block] = []
        current_size = 0
        ATOMIC = {"code", "table", "yaml_front_matter", "blockquote"}

        for block in blocks:
            bsize = len(block.content)

            if block.type in ATOMIC:
                # Flush if adding would exceed max (but always include atomic blocks)
                if current and current_size + bsize > self.max_chunk_size:
                    chunks.append(self._make_chunk(current))
                    current, current_size = [], 0
                current.append(block)
                current_size += bsize
                # Flush after atomic if over max
                if current_size >= self.max_chunk_size:
                    chunks.append(self._make_chunk(current))
                    current, current_size = [], 0
                continue

            if current_size + bsize > self.max_chunk_size and current:
                chunks.append(self._make_chunk(current))
                current, current_size = [], 0

            current.append(block)
            current_size += bsize

        if current:
            chunks.append(self._make_chunk(current))

        return self._merge_small_chunks(chunks)

    def _make_chunk(self, blocks: List[_Block]) -> Chunk:
        text = "\n\n".join(b.content for b in blocks)
        section_title = blocks[-1].section_title or blocks[0].section_title
        content_types = list(set(b.type for b in blocks))
        return Chunk(
            text=text,
            section_title=section_title,
            start_line=blocks[0].start_line,
            end_line=blocks[-1].end_line,
            content_types=content_types,
        )

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        merged: List[Chunk] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if len(c.text) < self.min_chunk_size and i + 1 < len(chunks):
                nxt = chunks[i + 1]
                if len(c.text) + len(nxt.text) <= self.max_chunk_size * 2:
                    merged.append(Chunk(
                        text=c.text + "\n\n" + nxt.text,
                        section_title=nxt.section_title or c.section_title,
                        start_line=c.start_line,
                        end_line=nxt.end_line,
                        content_types=list(
                            set(c.content_types + nxt.content_types)),
                    ))
                    i += 2
                    continue
            merged.append(c)
            i += 1
        return [c for c in merged if c.text.strip()]
