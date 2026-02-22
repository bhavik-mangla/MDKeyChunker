"""
Pass-keys-forward linking algorithm.

Implements intelligent key propagation across chunks using hybrid scoring.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import logging

from ..models.chunk import Chunk, ChunkMetadata, Entity, NumericValue
from ..utils.config import Config


@dataclass
class RollingKey:
    """Represents a key in the rolling memory."""
    name: str
    key_type: str  # entity, keyword, numeric
    last_seen_pos: int
    confidence: float
    metadata: Dict = None  # Additional metadata (entity type, etc.)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KeyLinker:
    """
    Implements pass-keys-forward algorithm for chunk linking.
    
    Maintains rolling memory of keys and scores potential linkages based on:
    - Entity overlap
    - Keyword overlap
    - Numeric value overlap
    - Proximity
    - Referential signals
    """
    
    # Referential cue patterns
    REFERENTIAL_PATTERNS = [
        r'\b(this|these|that|those)\b',
        r'\b(above|below|previous|following|earlier|later)\b',
        r'\b(aforementioned|mentioned|described|discussed)\b',
        r'\b(such|said)\b',
        r'\b(refer|reference|referring)\s+to\b',
    ]
    
    def __init__(self, config: Config):
        """Initialize key linker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rolling_keys: Dict[str, RollingKey] = {}
        
        # Compile referential patterns
        self.ref_patterns = [re.compile(p, re.IGNORECASE) for p in self.REFERENTIAL_PATTERNS]
    
    def reset(self) -> None:
        """Reset rolling memory."""
        self.rolling_keys = {}
    
    def process_chunks(
        self,
        chunks: List[Chunk],
        llm_client: Optional[object] = None,
        llm_budget: int = 0
    ) -> List[Chunk]:
        """
        Process chunks with pass-keys-forward linking.
        
        Args:
            chunks: List of chunks with metadata
            llm_client: Optional LLM client for ambiguity resolution
            llm_budget: Remaining LLM call budget
            
        Returns:
            Chunks with related_keys populated
        """
        if not self.config.pass_keys_forward:
            return chunks
        
        self.reset()
        llm_calls_used = 0
        
        # Forward pass
        for i, chunk in enumerate(chunks):
            if not chunk.metadata:
                continue
            
            # Extract keys from current chunk
            current_keys = self._extract_keys_from_chunk(chunk.metadata)
            
            # Score and match against rolling keys
            matched_keys, used_llm = self._match_keys(
                chunk.metadata,
                current_keys,
                i,
                llm_client if llm_calls_used < llm_budget else None
            )
            
            if used_llm:
                llm_calls_used += 1
            
            # Update chunk metadata
            chunk.metadata.related_keys = matched_keys
            
            # Update rolling keys with current chunk's keys
            self._update_rolling_keys(current_keys, i)
        
        # Backward pass if enabled
        if self.config.backward_pass:
            self._backward_pass(chunks, llm_client, llm_budget - llm_calls_used)
        
        self.logger.info(f"Pass-keys-forward complete. Used {llm_calls_used} LLM calls.")
        return chunks
    
    def _extract_keys_from_chunk(self, metadata: ChunkMetadata) -> List[RollingKey]:
        """Extract keys from a chunk's metadata."""
        keys: List[RollingKey] = []
        
        # Entity keys
        for entity in metadata.entities:
            if entity.name.lower() not in self.config.entity_blacklist:
                keys.append(RollingKey(
                    name=entity.name,
                    key_type='entity',
                    last_seen_pos=metadata.position_index,
                    confidence=1.0,
                    metadata={'entity_type': entity.type}
                ))
        
        # Keyword keys
        for keyword in metadata.keywords:
            if keyword.lower() not in self.config.entity_blacklist:
                keys.append(RollingKey(
                    name=keyword,
                    key_type='keyword',
                    last_seen_pos=metadata.position_index,
                    confidence=0.8
                ))
        
        # Numeric keys (store as string representation with separator)
        for num_val in metadata.numeric_values:
            key_name = f"{num_val.value}|{num_val.unit or 'none'}"
            keys.append(RollingKey(
                name=key_name,
                key_type='numeric',
                last_seen_pos=metadata.position_index,
                confidence=0.7,
                metadata={'value': num_val.value, 'unit': num_val.unit}
            ))
        
        return keys
    
    def _match_keys(
        self,
        metadata: ChunkMetadata,
        current_keys: List[RollingKey],
        position: int,
        llm_client: Optional[object]
    ) -> Tuple[List[str], bool]:
        """
        Match current chunk against rolling keys.
        
        Returns:
            Tuple of (matched_key_names, used_llm)
        """
        if not self.rolling_keys:
            return [], False
        
        matched: List[str] = []
        ambiguous: List[Tuple[str, float]] = []
        used_llm = False
        
        # Build sets for fast lookup
        current_entity_names = {e.name.lower() for e in metadata.entities}
        current_keywords = {k.lower() for k in metadata.keywords}
        current_numeric_strs = {f"{nv.value}|{nv.unit or 'none'}" for nv in metadata.numeric_values}
        
        for key_name, rolling_key in self.rolling_keys.items():
            score = self._compute_key_score(
                rolling_key,
                current_entity_names,
                current_keywords,
                current_numeric_strs,
                metadata.cleaned_text,
                position
            )
            
            if score >= self.config.threshold_accept:
                matched.append(key_name)
                self.logger.debug(
                    f"Accepted key '{key_name}' for chunk {position} (score={score:.3f})"
                )
            elif score >= self.config.threshold_reject:
                ambiguous.append((key_name, score))
                self.logger.debug(
                    f"Ambiguous key '{key_name}' for chunk {position} (score={score:.3f})"
                )
        
        # Resolve ambiguous keys with LLM if available
        if ambiguous and llm_client:
            llm_matched = self._resolve_with_llm(
                metadata,
                ambiguous,
                llm_client
            )
            matched.extend(llm_matched)
            used_llm = True
        
        return matched, used_llm
    
    def _compute_key_score(
        self,
        rolling_key: RollingKey,
        current_entities: Set[str],
        current_keywords: Set[str],
        current_numerics: Set[str],
        text: str,
        position: int
    ) -> float:
        """
        Compute matching score for a rolling key.
        
        Score = w1*entity_overlap + w2*keyword_overlap + w3*proximity + w4*ref_signal
        """
        score = 0.0
        
        # Entity overlap
        if rolling_key.key_type == 'entity':
            if rolling_key.name.lower() in current_entities:
                score += self.config.entity_weight
        
        # Keyword overlap
        if rolling_key.key_type == 'keyword':
            if rolling_key.name.lower() in current_keywords:
                score += self.config.keyword_weight
        
        # Numeric overlap
        if rolling_key.key_type == 'numeric':
            if rolling_key.name in current_numerics:
                score += self.config.keyword_weight * 0.8  # Slightly lower weight
        
        # Proximity (decay with distance)
        distance = position - rolling_key.last_seen_pos
        proximity_score = 1.0 / (1.0 + distance)
        score += self.config.proximity_weight * proximity_score
        
        # Referential signal
        ref_signal = self._detect_referential_cues(text)
        if ref_signal:
            score += self.config.ref_weight
        
        # Scale by key confidence
        score *= rolling_key.confidence
        
        return score
    
    def _detect_referential_cues(self, text: str) -> bool:
        """Detect if text contains referential cues."""
        for pattern in self.ref_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _resolve_with_llm(
        self,
        metadata: ChunkMetadata,
        ambiguous_keys: List[Tuple[str, float]],
        llm_client: object
    ) -> List[str]:
        """
        Resolve ambiguous key matches using LLM.
        
        Args:
            metadata: Current chunk metadata
            ambiguous_keys: List of (key_name, score) tuples
            llm_client: LLM client with resolve_key_matches method
            
        Returns:
            List of matched key names
        """
        try:
            # Call LLM (interface defined in llm module)
            result = llm_client.resolve_key_matches(
                chunk_text=metadata.cleaned_text,
                section_title=metadata.section_title,
                candidate_keys=[k for k, _ in ambiguous_keys],
                scores={k: s for k, s in ambiguous_keys}
            )
            
            if result and 'matched' in result:
                return result['matched']
        
        except Exception as e:
            self.logger.warning(f"LLM key resolution failed: {e}")
        
        return []
    
    def _update_rolling_keys(self, current_keys: List[RollingKey], position: int) -> None:
        """Update rolling keys memory with current chunk's keys."""
        for key in current_keys:
            # Normalize key name
            key_id = f"{key.key_type}:{key.name}"
            
            if key_id in self.rolling_keys:
                # Update existing key
                self.rolling_keys[key_id].last_seen_pos = position
                self.rolling_keys[key_id].confidence = max(
                    self.rolling_keys[key_id].confidence,
                    key.confidence
                )
            else:
                # Add new key
                self.rolling_keys[key_id] = key
        
        # Prune old keys if exceeding memory limit
        if len(self.rolling_keys) > self.config.max_keys_memory:
            # Remove oldest keys (smallest last_seen_pos)
            sorted_keys = sorted(
                self.rolling_keys.items(),
                key=lambda x: x[1].last_seen_pos,
                reverse=True
            )
            self.rolling_keys = dict(sorted_keys[:self.config.max_keys_memory])
    
    def _backward_pass(
        self,
        chunks: List[Chunk],
        llm_client: Optional[object],
        llm_budget: int
    ) -> None:
        """
        Perform backward pass to catch late references.
        
        Processes chunks in reverse order.
        """
        self.reset()
        llm_calls_used = 0
        
        for i in range(len(chunks) - 1, -1, -1):
            chunk = chunks[i]
            if not chunk.metadata:
                continue
            
            current_keys = self._extract_keys_from_chunk(chunk.metadata)
            
            # Match (in reverse)
            matched_keys, used_llm = self._match_keys(
                chunk.metadata,
                current_keys,
                i,
                llm_client if llm_calls_used < llm_budget else None
            )
            
            if used_llm:
                llm_calls_used += 1
            
            # Merge with forward pass results (union)
            existing = set(chunk.metadata.related_keys)
            combined = list(existing.union(set(matched_keys)))
            chunk.metadata.related_keys = combined
            
            # Update rolling keys
            self._update_rolling_keys(current_keys, i)
        
        self.logger.info(f"Backward pass complete. Used {llm_calls_used} LLM calls.")
