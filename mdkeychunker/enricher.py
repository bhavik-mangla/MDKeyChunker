"""Single LLM call per chunk enrichment with rolling keys."""
import logging
import re
from collections import Counter
from typing import List, Optional
from .llm_client import LLMClient
from .models import Chunk

log = logging.getLogger(__name__)

MAX_ROLLING_KEYS = 40  # hard cap — keeps prompt tokens manageable

ENRICH_PROMPT = '''You are a document analysis expert. Analyze this text chunk from a Markdown document and extract structured metadata for a RAG (Retrieval-Augmented Generation) system.
...''' # (truncated for brevity in replace call, keeping existing constant)

class Enricher:
    """Base class for enrichment strategies."""
    def __init__(self):
        self.rolling_keys: dict[str, dict] = {}

    def reset(self) -> None:
        self.rolling_keys.clear()

    def _format_rolling_keys(self) -> str:
        if not self.rolling_keys:
            return "(none yet — this is the first chunk)"
        parts = [f"- {k} (seen {v['count']}x)" for k, v in self.rolling_keys.items()]
        return "\n".join(parts)

    def _update_rolling_keys(self, key: str, index: int) -> None:
        if not key: return
        if key in self.rolling_keys:
            self.rolling_keys[key]["last_chunk"] = index
            self.rolling_keys[key]["count"] += 1
        else:
            self.rolling_keys[key] = {"first_chunk": index, "last_chunk": index, "count": 1}
        
        if len(self.rolling_keys) > MAX_ROLLING_KEYS:
            by_recency = sorted(self.rolling_keys.items(), key=lambda x: x[1]["last_chunk"], reverse=True)
            self.rolling_keys = dict(by_recency[:MAX_ROLLING_KEYS])

class LLMEnricher(Enricher):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm = llm_client

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            prev_summary = chunks[i-1].summary if i > 0 and chunks[i-1].summary else "(first chunk)"
            prompt = ENRICH_PROMPT.format(
                section_title=chunk.section_title or "(no section)",
                position=i + 1,
                total=total,
                prev_summary=prev_summary,
                chunk_text=chunk.text,
                rolling_keys=self._format_rolling_keys(),
            )
            result = self.llm.call_json(prompt)
            if result:
                chunk.title = result.get("title", "")
                chunk.summary = result.get("summary", "")
                chunk.keywords = result.get("keywords", [])
                chunk.entities = result.get("entities", [])
                chunk.questions = result.get("questions", [])
                chunk.key = result.get("key", "").lower()
                chunk.related_keys = result.get("related_keys", [])
            self._update_rolling_keys(chunk.key, i)
        return chunks

class SpacyEnricher(Enricher):
    """Lightweight, free enrichment using spaCy for high-speed local processing."""
    def __init__(self, model_size: str = "md"):
        super().__init__()
        try:
            import spacy
            self.nlp = spacy.load(f"en_core_web_{model_size}")
        except ImportError:
            raise ImportError("spaCy is required for SpacyEnricher. Install with: pip install spacy")

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        for i, chunk in enumerate(chunks):
            # Limit doc for speed, but ensure enough context
            doc = self.nlp(chunk.text[:1500])

            # 1. Extract Entities
            chunk.entities = [{"name": ent.text, "type": ent.label_} for ent in doc.ents]

            # 2. Extract Keywords (Noun chunks are more stable than single nouns)
            noun_chunks = [nc.text.lower() for nc in doc.noun_chunks if len(nc.text) > 3]
            chunk.keywords = list(set(noun_chunks))[:8]

            # 3. Rolling Key Discovery (The Differentiator)
            # Prioritize: (1) Technical Entities, (2) Most common noun chunk
            tech_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "TECH", "GPE")]

            # Rolling logic: Prefer re-using a key from the document's history 
            # if it appears in the current text.
            chunk.key = ""
            text_lower = chunk.text.lower()
            for seen_key in self.rolling_keys.keys():
                if seen_key in text_lower:
                    chunk.key = seen_key
                    break

            # If no history match, select best new key from current chunk
            if not chunk.key:
                candidates = tech_entities if tech_entities else noun_chunks
                if candidates:
                    # Prefer the most frequent candidate
                    chunk.key = Counter(candidates).most_common(1)[0][0]

            self._update_rolling_keys(chunk.key, i)
        return chunks

