"""Single LLM call per chunk enrichment with rolling keys."""
import logging
from .llm_client import LLMClient

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
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        # key_name → {first_chunk, last_chunk, count}
        self.rolling_keys: dict[str, dict] = {}

    def reset(self) -> None:
        """Clear rolling keys — call between documents to prevent key leakage."""
        self.rolling_keys.clear()

    def _format_rolling_keys(self) -> str:
        """Format rolling keys for the prompt with recency counts."""
        if not self.rolling_keys:
            return "(none yet — this is the first chunk)"
        parts = []
        for key_name, info in self.rolling_keys.items():
            parts.append(f"- {key_name} (seen {info['count']}x)")
        return "\n".join(parts)

    def _prune_rolling_keys(self) -> None:
        """Evict least-recently-seen keys when over the cap."""
        if len(self.rolling_keys) <= MAX_ROLLING_KEYS:
            return
        by_recency = sorted(
            self.rolling_keys.items(),
            key=lambda x: x[1]["last_chunk"],
            reverse=True,
        )
        self.rolling_keys = dict(by_recency[:MAX_ROLLING_KEYS])

    def enrich_chunks(self, chunks: list) -> list:
        """Enrich all chunks with a single LLM call each, maintaining rolling keys."""
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            prev_summary = (
                chunks[i - 1].summary if i > 0 and chunks[i - 1].summary
                else "(first chunk)"
            )

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

            if result:
                chunk.title = result.get("title", "")
                chunk.summary = result.get("summary", "")
                chunk.keywords = result.get("keywords", [])
                chunk.entities = result.get("entities", [])
                chunk.questions = result.get("questions", [])
                chunk.key = result.get("key", "")
                chunk.related_keys = result.get("related_keys", [])
            else:
                log.warning("LLM enrichment failed for chunk %d", i)

            # Update rolling keys
            if chunk.key:
                if chunk.key in self.rolling_keys:
                    self.rolling_keys[chunk.key]["last_chunk"] = i
                    self.rolling_keys[chunk.key]["count"] += 1
                else:
                    self.rolling_keys[chunk.key] = {
                        "first_chunk": i,
                        "last_chunk": i,
                        "count": 1,
                    }
                self._prune_rolling_keys()

        return chunks
