"""Keyword extraction using RAKE."""

from typing import List
import logging
import re

from ..utils.config import Config


class KeywordExtractor:
    """Extract keywords from text using RAKE algorithm."""
    
    def __init__(self, config: Config):
        """Initialize keyword extractor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rake = None
        self._load_rake()
    
    def _load_rake(self) -> None:
        """Load RAKE extractor."""
        try:
            from rake_nltk import Rake
            self.rake = Rake()
            self.logger.info("RAKE keyword extractor loaded")
        except Exception as e:
            self.logger.warning(f"Failed to load RAKE: {e}")
            self.rake = None
    
    def extract(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keyword strings
        """
        if not self.rake:
            # Fallback: simple frequency-based extraction
            return self._fallback_extraction(text, max_keywords)
        
        try:
            self.rake.extract_keywords_from_text(text)
            keywords = self.rake.get_ranked_phrases()
            
            # Filter and clean
            cleaned = []
            for kw in keywords[:max_keywords * 2]:  # Get more, then filter
                # Skip very long phrases
                if len(kw.split()) > 4:
                    continue
                # Skip very short
                if len(kw) < 3:
                    continue
                # Skip if all digits
                if kw.replace(' ', '').isdigit():
                    continue
                cleaned.append(kw)
                
                if len(cleaned) >= max_keywords:
                    break
            
            return cleaned
        
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return self._fallback_extraction(text, max_keywords)
    
    def _fallback_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Fallback keyword extraction using simple heuristics."""
        # Extract capitalized phrases and frequent multi-word terms
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Count frequencies
        freq: dict = {}
        for word in words:
            word_lower = word.lower()
            if len(word_lower) >= 3:  # Min length
                freq[word_lower] = freq.get(word_lower, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [kw for kw, _ in sorted_keywords[:max_keywords]]
