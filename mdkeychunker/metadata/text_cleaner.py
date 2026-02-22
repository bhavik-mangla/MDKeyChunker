"""Text cleaning utilities."""

import re
from typing import Tuple, List
import logging

from ..utils.config import Config


class TextCleaner:
    """Clean and normalize text while preserving character mapping."""
    
    def __init__(self, config: Config):
        """Initialize text cleaner."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def clean(self, text: str, preserve_mapping: bool = True) -> Tuple[str, List[int]]:
        """
        Clean text and optionally return character mapping.
        
        Args:
            text: Original text
            preserve_mapping: If True, return mapping from cleaned to original indices
            
        Returns:
            Tuple of (cleaned_text, char_map)
            char_map[i] gives the original position of character i in cleaned text
        """
        if not text:
            return "", []
        
        cleaned = text
        char_map: List[int] = list(range(len(text))) if preserve_mapping else []
        
        # Step 1: Normalize whitespace (but preserve newlines for structure)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces/tabs to single space
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)  # Multiple newlines to double newline
        
        # Step 2: Normalize quotes
        cleaned = self._normalize_quotes(cleaned)
        
        # Step 3: Normalize dashes
        cleaned = self._normalize_dashes(cleaned)
        
        # Step 4: Fix common OCR errors
        cleaned = self._fix_ocr_noise(cleaned)
        
        # Step 5: Remove leading/trailing whitespace per line
        lines = cleaned.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        cleaned = '\n'.join(cleaned_lines)
        
        # Rebuild character mapping if needed
        if preserve_mapping:
            char_map = self._build_char_map(text, cleaned)
        
        return cleaned, char_map
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote characters."""
        # Smart quotes to regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('«', '"').replace('»', '"')
        return text
    
    def _normalize_dashes(self, text: str) -> str:
        """Normalize various dash characters."""
        # Em dash, en dash to regular dash
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('‐', '-')  # Hyphen variants
        return text
    
    def _fix_ocr_noise(self, text: str) -> str:
        """Fix common OCR errors."""
        # Common OCR mistakes
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            '¦': '|',
            '¬': '-',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _build_char_map(self, original: str, cleaned: str) -> List[int]:
        """
        Build character mapping from cleaned to original text.
        
        This is a simplified version. A production implementation would use
        a more sophisticated alignment algorithm.
        """
        # Simple approach: assume mostly 1-to-1 mapping with some deletions
        char_map: List[int] = []
        orig_idx = 0
        
        for clean_idx, clean_char in enumerate(cleaned):
            # Find next matching character in original
            while orig_idx < len(original):
                if self._chars_match(original[orig_idx], clean_char):
                    char_map.append(orig_idx)
                    orig_idx += 1
                    break
                orig_idx += 1
            else:
                # No match found, use last position
                char_map.append(min(orig_idx, len(original) - 1))
        
        return char_map
    
    def _chars_match(self, orig_char: str, clean_char: str) -> bool:
        """Check if characters match after normalization."""
        # Normalize both for comparison
        orig_norm = self._normalize_quotes(self._normalize_dashes(orig_char))
        clean_norm = clean_char
        
        return orig_norm == clean_norm or orig_norm.lower() == clean_norm.lower()
