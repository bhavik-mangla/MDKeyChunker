"""Entity extraction using spaCy."""

from typing import List, Optional, Set
import logging

from ..models.chunk import Entity
from ..utils.config import Config


class EntityExtractor:
    """Extract named entities from text using spaCy."""
    
    def __init__(self, config: Config):
        """Initialize entity extractor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.blacklist: Set[str] = set(word.lower() for word in config.entity_blacklist)
        
        if config.entity_extraction_mode == "spacy":
            self._load_spacy_model()
    
    def _load_spacy_model(self) -> None:
        """Load spaCy model lazily."""
        try:
            import spacy
            self.nlp = spacy.load(self.config.spacy_model)
            self.logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def extract(self, text: str, cleaned_text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Original text
            cleaned_text: Cleaned text (used for extraction)
            
        Returns:
            List of Entity objects with span positions in original text
        """
        if self.config.entity_extraction_mode == "none" or not self.nlp:
            return []
        
        entities: List[Entity] = []
        
        try:
            doc = self.nlp(cleaned_text)
            
            for ent in doc.ents:
                # Filter by blacklist if strict mode
                if self.config.strict_entity_mode:
                    if ent.text.lower() in self.blacklist:
                        continue
                    # Filter out very short or very common entities
                    if len(ent.text) < 2:
                        continue
                    # Filter out pure numbers (handled separately)
                    if ent.text.isdigit():
                        continue
                
                # Map back to original text positions
                # For simplicity, we use cleaned_text positions
                # In production, would need character mapping
                entities.append(Entity(
                    name=ent.text,
                    type=ent.label_,
                    span_start=ent.start_char,
                    span_end=ent.end_char
                ))
        
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    def extract_with_mapping(self, text: str, cleaned_text: str, char_map: List[int]) -> List[Entity]:
        """
        Extract entities with character mapping from cleaned to original text.
        
        Args:
            text: Original text
            cleaned_text: Cleaned text
            char_map: Mapping from cleaned_text indices to original text indices
            
        Returns:
            List of entities with positions in original text
        """
        if self.config.entity_extraction_mode == "none" or not self.nlp:
            return []
        
        entities: List[Entity] = []
        
        try:
            doc = self.nlp(cleaned_text)
            
            for ent in doc.ents:
                # Apply filters
                if self.config.strict_entity_mode:
                    if ent.text.lower() in self.blacklist:
                        continue
                    if len(ent.text) < 2 or ent.text.isdigit():
                        continue
                
                # Map positions back to original text
                try:
                    orig_start = char_map[ent.start_char] if ent.start_char < len(char_map) else ent.start_char
                    orig_end = char_map[ent.end_char - 1] + 1 if ent.end_char - 1 < len(char_map) else ent.end_char
                except IndexError:
                    orig_start = ent.start_char
                    orig_end = ent.end_char
                
                entities.append(Entity(
                    name=ent.text,
                    type=ent.label_,
                    span_start=orig_start,
                    span_end=orig_end
                ))
        
        except Exception as e:
            self.logger.error(f"Entity extraction with mapping failed: {e}")
        
        return entities
