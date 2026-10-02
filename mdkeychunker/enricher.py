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

**Section Path:** {section_title}
**Chunk Position:** {position} of {total} chunks
**Previous Chunk Summary:** {prev_summary}

**Chunk Text:**
{chunk_text}

**Rolling Keys (specific subtopics seen in previous chunks):**
{rolling_keys}

Extract the following in a single JSON response:

{{
  "title": "A short descriptive title for this chunk (3-8 words)",
  "summary": "A 1-2 sentence summary (30-60 words) capturing the key information. Do NOT just repeat the first sentence. Focus on what makes this chunk UNIQUE — what would a search engine snippet show?",
  "keywords": ["5-8 salient terms or phrases, domain-specific preferred"],
  "entities": [
    {{"name": "entity name", "type": "PERSON|ORG|LOC|TECH|CONCEPT|EVENT|METRIC"}}
  ],
  "questions": ["2-3 specific questions this chunk can answer"],
  "key": "The SPECIFIC subtopic that makes this chunk UNIQUE within the document. 2-5 words, lowercase. CRITICAL RULES: (1) Must DISTINGUISH this chunk from other chunks about the same broad topic. (2) Think: if someone asked what SPECIFIC ASPECT this chunk covers, what would you say? (3) Examples: admissions process, gradient descent optimization, oauth token flow, q3 revenue breakdown. (4) Two chunks should share a key ONLY if they cover the EXACT same specific aspect and would make a coherent single piece when combined. (5) REUSE a key from the rolling keys list if this chunk CONTINUES the same specific discussion. (6) A key should NOT be the document broad topic — it must be more specific than that. (7) NEVER use a 1-word key that could describe the whole document.",
  "related_keys": ["From the rolling keys above, pick 0-3 keys that this chunk DIRECTLY discusses or depends on. Err on the side of fewer. Ask: would a reader need to read the related-key chunk to understand THIS chunk? If not, do not include it. An empty list is perfectly fine."]
}}

Rules:
- "related_keys" must be a SUBSET of the rolling keys provided — only include genuinely relevant ones
- "entities" should include technical terms, proper nouns, and domain concepts with types: PERSON (people), ORG (organizations), LOC (locations), TECH (technologies/tools), CONCEPT (abstract concepts), EVENT (events/dates), METRIC (measurements/KPIs)
- "keywords" should be specific and domain-relevant (not generic words like "system", "data", "process")
- "questions" should be natural questions a user would ask that this chunk answers
- Return ONLY valid JSON, no extra text'''

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
            try:
                result = self.llm.call_json(prompt, max_tokens=1000)
            except Exception as e:
                log.warning("LLM call failed for chunk %d: %s", i, e)
                result = None

            if isinstance(result, dict):
                chunk.title = _as_str(result.get("title"))
                chunk.summary = _as_str(result.get("summary"))
                chunk.keywords = _as_str_list(result.get("keywords"))
                chunk.entities = _as_entities(result.get("entities"))
                chunk.questions = _as_str_list(result.get("questions"))
                chunk.key = _as_str(result.get("key"), MAX_KEY_LEN).lower()
                chunk.related_keys = _as_str_list(result.get("related_keys"))
            else:
                log.warning("LLM enrichment failed for chunk %d", i)
            self._update_rolling_keys(chunk.key, i)
        return chunks


MAX_FIELD_LEN = 1000  # caps runaway LLM output that would otherwise grow later prompts
MAX_KEY_LEN = 80


def _as_str(value, limit: int = MAX_FIELD_LEN) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _as_str_list(value) -> list:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return []
    return [s for s in (_as_str(v) for v in value) if s]


def _as_entities(value) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for e in value:
        if isinstance(e, dict) and _as_str(e.get("name")):
            out.append({"name": _as_str(e.get("name")), "type": _as_str(e.get("type")) or "CONCEPT"})
        elif isinstance(e, str) and e.strip():
            out.append({"name": _as_str(e), "type": "CONCEPT"})
    return out

class SpacyEnricher(Enricher):
    """Lightweight, free enrichment using spaCy for high-speed local processing."""
    def __init__(self, model_size: str = "md"):
        super().__init__()
        try:
            import spacy
            self.nlp = spacy.load(f"en_core_web_{model_size}")
        except ImportError:
            raise ImportError(
                "spaCy is required for SpacyEnricher. Install with: pip install 'mdkeychunker[spacy]'"
            )
        except OSError:
            raise OSError(
                f"spaCy model en_core_web_{model_size} is not installed. "
                f"Install with: python -m spacy download en_core_web_{model_size}"
            )

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
                if re.search(rf"\b{re.escape(seen_key)}\b", text_lower):
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

