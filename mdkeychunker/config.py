"""Configuration for the legacy ``Pipeline`` (0.3 API, deprecated in 0.4)."""
import os
import warnings
from dataclasses import dataclass
from typing import Optional

_ORPHAN_DEFAULT = 200


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
    # Restructuring. Off by default since 0.4 (was True): the v4 study found no retrieval
    # difference between key-based and similarity merges. None means "left at the default".
    merge_by_keys: Optional[bool] = None
    max_merged_size: int = 3000            # Max size after merging
    # Deprecated: when merging runs, a keyless chunk below this size gets its neighbour's
    # text prepended. It is not an LLM "context enrichment".
    min_orphan_size: int = _ORPHAN_DEFAULT
    # General; read by the CLI only, the library never configures logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.min_orphan_size != _ORPHAN_DEFAULT:
            warnings.warn("min_orphan_size is deprecated and will be removed in 0.6",
                          DeprecationWarning, stacklevel=3)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Build a Config from environment variables.

        Since 0.4 a ``.env`` file is read only when ``env_file`` is given (needs
        ``pip install "mdkeychunker[cli]"``); ``./.env`` is no longer loaded implicitly.
        """
        if env_file:
            try:
                from dotenv import load_dotenv
            except ImportError as e:
                raise ImportError(
                    'Reading an env file needs: pip install "mdkeychunker[cli]"') from e
            load_dotenv(env_file)
        elif os.path.exists(".env"):
            warnings.warn("Config.from_env() no longer reads ./.env (0.4); pass env_file='.env'",
                          DeprecationWarning, stacklevel=2)
        merge = os.getenv("MERGE_BY_KEYS")
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1500")),
            merge_by_keys=None if merge is None else merge.lower() in ("true", "1", "yes"),
            max_merged_size=int(os.getenv("MAX_MERGED_SIZE", "3000")),
            min_orphan_size=int(os.getenv("MIN_ORPHAN_SIZE", str(_ORPHAN_DEFAULT))),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
