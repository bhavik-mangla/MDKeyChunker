"""Main pipeline: chunk → enrich → restructure."""
import logging
from pathlib import Path
from .config import Config
from .chunker import MarkdownChunker
from .enricher import Enricher
from .llm_client import LLMClient
from .restructurer import Restructurer

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.chunker = MarkdownChunker(self.config)
        self.llm = LLMClient(self.config)
        self.enricher = Enricher(self.llm)
        self.restructurer = Restructurer(self.config)

    def process_file(self, filepath: str) -> list:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        content = path.read_text(encoding="utf-8")
        return self.process_text(content)

    def process_text(self, text: str) -> list:
        # Reset enricher state so rolling keys don't leak between documents
        self.enricher.reset()

        # Step 1: Structure-aware chunking
        chunks = self.chunker.chunk(text)
        log.info("Chunked into %d segments", len(chunks))

        # Step 2: LLM enrichment with rolling keys
        chunks = self.enricher.enrich_chunks(chunks)
        log.info("Enriched %d chunks", len(chunks))

        # Step 3: Key-based restructuring
        chunks = self.restructurer.restructure(chunks)
        log.info("Restructured to %d chunks", len(chunks))

        # Step 4: Set navigation + token counts
        self._finalize(chunks)
        return chunks

    def _finalize(self, chunks: list) -> None:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            def count_tokens(text: str) -> int: return len(enc.encode(text))
        except Exception:
            def count_tokens(text: str) -> int: return len(text) // 4  # rough estimate

        for i, c in enumerate(chunks):
            c.position_index = i
            c.generate_id()
            c.token_count = count_tokens(c.text)

        for i in range(len(chunks)):
            chunks[i].previous_chunk_id = chunks[i -
                                                 1].chunk_id if i > 0 else ""
            chunks[i].next_chunk_id = (
                chunks[i + 1].chunk_id if i < len(chunks) - 1 else ""
            )

    def save_jsonl(self, chunks: list, path: str) -> None:
        Path(path).write_text(
            "\n".join(c.to_json() for c in chunks), encoding="utf-8"
        )

    def save_summary(self, chunks: list, path: str) -> None:
        lines = [
            f"[{i + 1}] {c.title or c.section_title}: {c.summary}"
            for i, c in enumerate(chunks)
            if c.summary
        ]
        Path(path).write_text("\n".join(lines), encoding="utf-8")
