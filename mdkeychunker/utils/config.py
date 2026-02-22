"""Configuration management."""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for MDKeyChunker."""
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_call_budget_per_doc: int = 50
    
    # Entity Extraction
    entity_extraction_mode: str = "spacy"
    spacy_model: str = "en_core_web_sm"
    strict_entity_mode: bool = True
    entity_blacklist: List[str] = field(default_factory=lambda: [
        "system", "process", "data", "information", "document", "section",
        "chapter", "page", "example", "item", "thing", "way", "time",
        "year", "day", "case", "part", "point", "fact"
    ])
    
    # Chunking Parameters
    min_chunk_size: int = 100
    soft_max_chunk_size: int = 1000
    hard_max_chunk_size: int = 2000
    chunk_max_tokens: int = 512
    
    # Pass-Keys-Forward
    pass_keys_forward: bool = True
    backward_pass: bool = False
    max_keys_memory: int = 100
    
    # Scoring Weights
    entity_weight: float = 0.4
    keyword_weight: float = 0.3
    proximity_weight: float = 0.2
    ref_weight: float = 0.1
    
    # Thresholds
    threshold_accept: float = 0.6
    threshold_reject: float = 0.3
    
    # Optional Features
    enable_questions: bool = False
    duplicate_detection: bool = True
    remove_headers_footers: bool = True
    
    # Performance
    multiprocessing: bool = False
    max_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on')
        
        def get_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                return default
        
        def get_list(key: str, default: List[str]) -> List[str]:
            value = os.getenv(key, "")
            if not value:
                return default
            return [item.strip().lower() for item in value.split(',') if item.strip()]
        
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_call_budget_per_doc=get_int("LLM_CALL_BUDGET_PER_DOC", 50),
            
            entity_extraction_mode=os.getenv("ENTITY_EXTRACTION_MODE", "spacy"),
            spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm"),
            strict_entity_mode=get_bool("STRICT_ENTITY_MODE", True),
            entity_blacklist=get_list("ENTITY_BLACKLIST", ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]),
            
            min_chunk_size=get_int("MIN_CHUNK_SIZE", 100),
            soft_max_chunk_size=get_int("SOFT_MAX_CHUNK_SIZE", 1000),
            hard_max_chunk_size=get_int("HARD_MAX_CHUNK_SIZE", 2000),
            chunk_max_tokens=get_int("CHUNK_MAX_TOKENS", 512),
            
            pass_keys_forward=get_bool("PASS_KEYS_FORWARD", True),
            backward_pass=get_bool("BACKWARD_PASS", False),
            max_keys_memory=get_int("MAX_KEYS_MEMORY", 100),
            
            entity_weight=get_float("ENTITY_WEIGHT", 0.4),
            keyword_weight=get_float("KEYWORD_WEIGHT", 0.3),
            proximity_weight=get_float("PROXIMITY_WEIGHT", 0.2),
            ref_weight=get_float("REF_WEIGHT", 0.1),
            
            threshold_accept=get_float("THRESHOLD_ACCEPT", 0.6),
            threshold_reject=get_float("THRESHOLD_REJECT", 0.3),
            
            enable_questions=get_bool("ENABLE_QUESTIONS", False),
            duplicate_detection=get_bool("DUPLICATE_DETECTION", True),
            remove_headers_footers=get_bool("REMOVE_HEADERS_FOOTERS", True),
            
            multiprocessing=get_bool("MULTIPROCESSING", False),
            max_workers=get_int("MAX_WORKERS", 4),
            
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE") or None,
        )
