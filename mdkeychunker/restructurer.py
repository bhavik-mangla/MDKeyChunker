"""Restructure chunks by merging those with same/related keys."""
from collections import defaultdict
from .models import Chunk
from .config import Config


class Restructurer:
    def __init__(self, config: Config):
        self.config = config

    def restructure(self, chunks: list[Chunk]) -> list[Chunk]:
        """RAG-optimized restructuring: merge by key globally, handle orphans."""
        if not self.config.merge_by_keys or not chunks:
            return chunks

        # Phase 1: Group ALL chunks by key (globally, regardless of position)
        key_groups: dict[str, list[int]] = defaultdict(list)
        no_key: list[int] = []
        for i, chunk in enumerate(chunks):
            if chunk.key:
                key_groups[chunk.key].append(i)
            else:
                no_key.append(i)

        new_chunks: list[Chunk] = []

        # Phase 2: Merge same-key groups with bin packing (respects max_merged_size)
        for key, indices in key_groups.items():
            merged_bins: list[list[int]] = []
            current_bin: list[int] = [indices[0]]
            current_size = len(chunks[indices[0]].text)

            for idx in indices[1:]:
                candidate_size = len(chunks[idx].text)
                if current_size + candidate_size + 2 <= self.config.max_merged_size:
                    current_bin.append(idx)
                    current_size += candidate_size + 2  # +2 for "\n\n"
                else:
                    merged_bins.append(current_bin)
                    current_bin = [idx]
                    current_size = candidate_size
            merged_bins.append(current_bin)

            for bin_indices in merged_bins:
                new_chunks.append(self._merge_chunk_group(
                    [chunks[i] for i in bin_indices]
                ))

        # Phase 3: Handle keyless/orphan chunks
        for idx in no_key:
            chunk = chunks[idx]
            if len(chunk.text) < self.config.min_orphan_size:
                chunk = self._enrich_orphan_context(chunk, chunks, idx)
            new_chunks.append(chunk)

        # Phase 4: Sort by original position (start_line)
        new_chunks.sort(key=lambda c: c.start_line)
        return new_chunks

    def _merge_chunk_group(self, group: list[Chunk]) -> Chunk:
        """Merge a list of same-key chunks into one."""
        if len(group) == 1:
            return group[0]

        first = group[0]
        merged_text = group[0].text
        merged_entities = list(group[0].entities)
        merged_keywords = list(group[0].keywords)
        merged_questions = list(group[0].questions)
        merged_related = list(group[0].related_keys)
        merged_types = list(group[0].content_types)
        last_end_line = group[0].end_line

        for chunk in group[1:]:
            merged_text += "\n\n" + chunk.text
            merged_entities.extend(chunk.entities)
            merged_keywords.extend(chunk.keywords)
            merged_questions.extend(chunk.questions)
            merged_related.extend(chunk.related_keys)
            merged_types.extend(chunk.content_types)
            last_end_line = max(last_end_line, chunk.end_line)

        return Chunk(
            text=merged_text,
            section_title=first.section_title,
            content_types=list(set(merged_types)),
            start_line=first.start_line,
            end_line=last_end_line,
            title=first.title,
            summary=first.summary,
            keywords=list(set(merged_keywords)),
            entities=_dedupe_entities(merged_entities),
            questions=list(set(merged_questions)),
            key=first.key,
            related_keys=list(set(merged_related)),
        )

    def _enrich_orphan_context(self, chunk: Chunk, all_chunks: list[Chunk], idx: int) -> Chunk:
        """Prepend section/neighbor context to small orphan chunks for better RAG retrieval.

        We do NOT merge with neighbors (that would mix topics).
        Instead, we add a lightweight context header so the embedding has more signal.
        """
        context_parts = []

        if chunk.section_title:
            context_parts.append(f"[Section: {chunk.section_title}]")

        if idx > 0 and all_chunks[idx - 1].summary:
            context_parts.append(f"[Previous: {all_chunks[idx - 1].summary}]")
        if idx < len(all_chunks) - 1 and all_chunks[idx + 1].summary:
            context_parts.append(f"[Next: {all_chunks[idx + 1].summary}]")

        if context_parts:
            chunk.text = "\n".join(context_parts) + "\n\n" + chunk.text

        return chunk


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result = []
    for e in entities:
        key = (e.get("name", "").lower(), e.get("type", ""))
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result
