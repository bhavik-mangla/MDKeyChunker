"""Configuration for MDKeyChunker."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    # LLM
    llm_provider: str = "openai"           # openai | anthropic | openai_compatible
    llm_api_key: str = ""
    llm_base_url: str = ""                 # For Ollama/vLLM: http://localhost:11434/v1
    llm_model: str = "gpt-4o-mini"
    # Chunking
    min_chunk_size: int = 100              # Minimum chars per chunk
    max_chunk_size: int = 1500             # Soft max chars per chunk
    # Restructuring
    merge_by_keys: bool = True             # Enable key-based chunk merging
    max_merged_size: int = 3000            # Max size after merging
    # Below this, keyless chunks get context enrichment
    min_orphan_size: int = 200
    # General
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Config":
        load_dotenv(env_file or ".env")
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1500")),
            merge_by_keys=os.getenv(
                "MERGE_BY_KEYS", "true").lower() in ("true", "1", "yes"),
            max_merged_size=int(os.getenv("MAX_MERGED_SIZE", "3000")),
            min_orphan_size=int(os.getenv("MIN_ORPHAN_SIZE", "200")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
