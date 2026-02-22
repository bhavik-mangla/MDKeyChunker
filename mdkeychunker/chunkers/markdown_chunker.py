"""
Markdown structural chunker.

Splits Markdown documents following structure (headers, lists, tables, code blocks)
without breaking semantic units.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from ..models.chunk import Chunk
from ..utils.config import Config


@dataclass
class MarkdownBlock:
    """Represents a structural block in Markdown."""
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
        """Initialize the chunker with configuration."""
        self.config = config
        self.min_chunk_size = config.min_chunk_size
        self.soft_max_chunk_size = config.soft_max_chunk_size
        self.hard_max_chunk_size = config.hard_max_chunk_size
        self._header_stack: List[Tuple[int, str]] = []  # (level, title)

    def chunk(self, markdown_text: str) -> List[Chunk]:
        """
        Chunk a Markdown document into structural units.
        
        Args:
            markdown_text: Raw Markdown content
            
        Returns:
            List of Chunk objects preserving Markdown structure
        """
        # Parse into structural blocks
        blocks = self._parse_markdown_blocks(markdown_text)
        
        # Detect and remove repeating headers/footers if enabled
        if self.config.remove_headers_footers:
            blocks = self._remove_repeating_patterns(blocks)
        
        # Group blocks into chunks respecting size constraints
        chunks = self._group_blocks_into_chunks(blocks)
        
        # Deduplicate if enabled
        if self.config.duplicate_detection:
            chunks = self._deduplicate_chunks(chunks)
        
        return chunks

    def _parse_markdown_blocks(self, text: str) -> List[MarkdownBlock]:
        """Parse Markdown into structural blocks."""
        blocks: List[MarkdownBlock] = []
        lines = text.split('\n')
        i = 0
        current_section = ""
        
        while i < len(lines):
            line = lines[i]
            
            # YAML front matter
            if i == 0 and line.strip() == '---':
                block, lines_consumed = self._parse_yaml_front_matter(lines, i)
                if block:
                    blocks.append(block)
                    i += lines_consumed
                    continue
            
            # Code blocks (fenced)
            if line.strip().startswith('```') or line.strip().startswith('~~~'):
                block, lines_consumed = self._parse_code_block(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += lines_consumed
                continue
            
            # Headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update header stack
                self._update_header_stack(level, title)
                current_section = self._get_current_section()
                
                blocks.append(MarkdownBlock(
                    type='header',
                    content=line,
                    start_line=i,
                    end_line=i,
                    level=level,
                    section_title=current_section
                ))
                i += 1
                continue
            
            # Tables
            if self._is_table_line(line, lines, i):
                block, lines_consumed = self._parse_table(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += lines_consumed
                continue
            
            # Lists
            if self._is_list_line(line):
                block, lines_consumed = self._parse_list(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += lines_consumed
                continue
            
            # Blockquotes
            if line.strip().startswith('>'):
                block, lines_consumed = self._parse_blockquote(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += lines_consumed
                continue
            
            # Horizontal rules
            if re.match(r'^(\*\*\*+|---+|___+)\s*$', line):
                blocks.append(MarkdownBlock(
                    type='horizontal_rule',
                    content=line,
                    start_line=i,
                    end_line=i,
                    section_title=current_section
                ))
                i += 1
                continue
            
            # Paragraphs
            if line.strip():
                block, lines_consumed = self._parse_paragraph(lines, i)
                block.section_title = current_section
                blocks.append(block)
                i += lines_consumed
                continue
            
            # Empty lines (skip)
            i += 1
        
        return blocks

    def _parse_yaml_front_matter(self, lines: List[str], start: int) -> Tuple[Optional[MarkdownBlock], int]:
        """Parse YAML front matter."""
        # Handle UTF-8 BOM at start of file
        first_line = lines[start].lstrip('\ufeff').strip() if start == 0 else lines[start].strip()
        if not (start == 0 and first_line == '---'):
            return None, 0
        
        end = start + 1
        while end < len(lines) and lines[end].strip() != '---':
            end += 1
        
        if end >= len(lines):
            return None, 0
        
        content = '\n'.join(lines[start:end + 1])
        return MarkdownBlock(
            type='yaml_front_matter',
            content=content,
            start_line=start,
            end_line=end,
            section_title=""
        ), end - start + 1

    def _parse_code_block(self, lines: List[str], start: int) -> Tuple[MarkdownBlock, int]:
        """Parse a fenced code block."""
        fence = lines[start].strip()[:3]
        end = start + 1
        
        while end < len(lines):
            if lines[end].strip().startswith(fence):
                end += 1
                break
            end += 1
        
        content = '\n'.join(lines[start:end])
        return MarkdownBlock(
            type='code',
            content=content,
            start_line=start,
            end_line=end - 1,
            section_title=""
        ), end - start

    def _is_table_line(self, line: str, lines: List[str], idx: int) -> bool:
        """Check if line is part of a table."""
        # Simple heuristic: contains multiple | characters
        if '|' not in line:
            return False
        
        # Check if next line is a separator (e.g., |---|---|)
        if idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if re.match(r'^\s*\|?[\s:-]+\|[\s|:-]+\|?\s*$', next_line):
                return True
        
        # Or if previous line was a table
        if idx > 0:
            prev_line = lines[idx - 1]
            if '|' in prev_line or re.match(r'^\s*\|?[\s:-]+\|[\s|:-]+\|?\s*$', prev_line):
                return True
        
        return False

    def _parse_table(self, lines: List[str], start: int) -> Tuple[MarkdownBlock, int]:
        """Parse a Markdown table."""
        end = start
        
        # Continue while lines contain |
        while end < len(lines) and ('|' in lines[end] or not lines[end].strip()):
            if not lines[end].strip():
                break
            end += 1
        
        content = '\n'.join(lines[start:end])
        return MarkdownBlock(
            type='table',
            content=content,
            start_line=start,
            end_line=end - 1,
            section_title=""
        ), end - start

    def _is_list_line(self, line: str) -> bool:
        """Check if line is a list item."""
        stripped = line.strip()
        # Unordered list
        if re.match(r'^[\*\-\+]\s+', stripped):
            return True
        # Ordered list
        if re.match(r'^\d+\.\s+', stripped):
            return True
        return False

    def _parse_list(self, lines: List[str], start: int) -> Tuple[MarkdownBlock, int]:
        """Parse a list (ordered or unordered)."""
        end = start
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        
        end += 1
        while end < len(lines):
            line = lines[end]
            if not line.strip():
                # Check if next non-empty line continues the list
                peek = end + 1
                while peek < len(lines) and not lines[peek].strip():
                    peek += 1
                if peek < len(lines) and (self._is_list_line(lines[peek]) or 
                                          len(lines[peek]) - len(lines[peek].lstrip()) > base_indent):
                    end += 1
                    continue
                break
            
            # Continuation (indented) or new list item
            indent = len(line) - len(line.lstrip())
            if self._is_list_line(line) or indent > base_indent:
                end += 1
            else:
                break
        
        content = '\n'.join(lines[start:end])
        return MarkdownBlock(
            type='list',
            content=content,
            start_line=start,
            end_line=end - 1,
            section_title=""
        ), end - start

    def _parse_blockquote(self, lines: List[str], start: int) -> Tuple[MarkdownBlock, int]:
        """Parse a blockquote."""
        end = start
        
        while end < len(lines) and lines[end].strip().startswith('>'):
            end += 1
        
        content = '\n'.join(lines[start:end])
        return MarkdownBlock(
            type='blockquote',
            content=content,
            start_line=start,
            end_line=end - 1,
            section_title=""
        ), end - start

    def _parse_paragraph(self, lines: List[str], start: int) -> Tuple[MarkdownBlock, int]:
        """Parse a paragraph."""
        end = start
        
        while end < len(lines):
            line = lines[end]
            
            # Stop at empty line (but consume first line if it's the start)
            if not line.strip():
                if end == start:
                    end += 1  # Consume the empty line
                break
            
            # For first line, always include it
            if end == start:
                end += 1
                continue
            
            # Stop at other block types (only check after first line)
            if (line.strip().startswith('#') or 
                line.strip().startswith('```') or 
                line.strip().startswith('~~~') or
                self._is_list_line(line) or
                line.strip().startswith('>') or
                '|' in line):
                break
            
            end += 1
        
        content = '\n'.join(lines[start:end])
        return MarkdownBlock(
            type='paragraph',
            content=content,
            start_line=start,
            end_line=end - 1,
            section_title=""
        ), end - start

    def _update_header_stack(self, level: int, title: str) -> None:
        """Update the header hierarchy stack."""
        # Remove headers at same or deeper level
        self._header_stack = [(l, t) for l, t in self._header_stack if l < level]
        self._header_stack.append((level, title))

    def _get_current_section(self) -> str:
        """Get the current section path from header stack."""
        if not self._header_stack:
            return ""
        return " > ".join(title for _, title in self._header_stack)

    def _group_blocks_into_chunks(self, blocks: List[MarkdownBlock]) -> List[Chunk]:
        """Group blocks into chunks respecting size constraints."""
        chunks: List[Chunk] = []
        current_chunk_blocks: List[MarkdownBlock] = []
        current_size = 0
        
        for block in blocks:
            block_size = len(block.content)
            
            # Never split atomic blocks (code, table, yaml_front_matter)
            if block.type in ['code', 'table', 'yaml_front_matter', 'blockquote']:
                # If adding this block exceeds hard max and we have content, flush first
                if current_chunk_blocks and current_size + block_size > self.hard_max_chunk_size:
                    chunks.append(self._create_chunk_from_blocks(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_size = 0
                
                # Add the block (even if it exceeds hard_max by itself)
                current_chunk_blocks.append(block)
                current_size += block_size
                
                # Flush if we've exceeded soft max
                if current_size >= self.soft_max_chunk_size:
                    chunks.append(self._create_chunk_from_blocks(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_size = 0
                
                continue
            
            # For other blocks, try to respect soft_max
            if current_size + block_size > self.soft_max_chunk_size and current_chunk_blocks:
                # Flush current chunk
                chunks.append(self._create_chunk_from_blocks(current_chunk_blocks))
                current_chunk_blocks = []
                current_size = 0
            
            current_chunk_blocks.append(block)
            current_size += block_size
        
        # Flush remaining blocks
        if current_chunk_blocks:
            chunks.append(self._create_chunk_from_blocks(current_chunk_blocks))
        
        # Merge undersized chunks
        chunks = self._merge_small_chunks(chunks)
        
        return chunks

    def _create_chunk_from_blocks(self, blocks: List[MarkdownBlock]) -> Chunk:
        """Create a Chunk from a list of blocks."""
        if not blocks:
            return Chunk(text="", section_title="", content_types=[])
        
        text = '\n\n'.join(block.content for block in blocks)
        section_title = blocks[-1].section_title or blocks[0].section_title
        start_line = blocks[0].start_line
        end_line = blocks[-1].end_line
        
        # Collect unique content types
        content_types = list(set(block.type for block in blocks))
        
        return Chunk(
            text=text,
            section_title=section_title,
            start_line=start_line,
            end_line=end_line,
            content_types=content_types
        )

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than min_chunk_size with neighbors."""
        if not chunks:
            return chunks
        
        merged: List[Chunk] = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If chunk is too small and not the last one, try to merge with next
            if len(current.text) < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # Merge if combined size doesn't exceed hard max
                combined_size = len(current.text) + len(next_chunk.text)
                if combined_size <= self.hard_max_chunk_size:
                    merged_text = current.text + '\n\n' + next_chunk.text
                    merged_types = list(set(current.content_types + next_chunk.content_types))
                    
                    merged_chunk = Chunk(
                        text=merged_text,
                        section_title=next_chunk.section_title or current.section_title,
                        start_line=current.start_line,
                        end_line=next_chunk.end_line,
                        content_types=merged_types
                    )
                    merged.append(merged_chunk)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        # Filter out empty chunks
        return [c for c in merged if c.text.strip()]

    def _remove_repeating_patterns(self, blocks: List[MarkdownBlock]) -> List[MarkdownBlock]:
        """Detect and remove repeating headers or footers."""
        if len(blocks) < 3:
            return blocks
        
        # Simple heuristic: if same content appears at regular intervals, it might be a header/footer
        content_positions: Dict[str, List[int]] = {}
        
        for i, block in enumerate(blocks):
            normalized = block.content.strip()
            if normalized:
                if normalized not in content_positions:
                    content_positions[normalized] = []
                content_positions[normalized].append(i)
        
        # Find patterns that repeat (appear 3+ times)
        repeating_indices: Set[int] = set()
        for content, positions in content_positions.items():
            if len(positions) >= 3 and len(content) < 200:  # Short repeated content
                # Check if intervals are regular
                intervals = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
                avg_interval = sum(intervals) / len(intervals)
                
                # If intervals are similar (within 20% variance), mark as repeating
                if all(abs(interval - avg_interval) / avg_interval < 0.2 for interval in intervals):
                    repeating_indices.update(positions)
        
        # Remove repeating blocks
        filtered = [block for i, block in enumerate(blocks) if i not in repeating_indices]
        return filtered if filtered else blocks

    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks based on content hash."""
        if not self.config.duplicate_detection:
            return chunks
        
        seen_hashes: Set[str] = set()
        unique_chunks: List[Chunk] = []
        
        for chunk in chunks:
            # Hash the normalized content
            content_hash = hashlib.md5(chunk.text.strip().encode('utf-8'), usedforsecurity=False).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
