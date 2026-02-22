"""Numeric value extraction and normalization."""

import re
from typing import List, Optional, Tuple
import logging

from ..models.chunk import NumericValue
from ..utils.config import Config


class NumericExtractor:
    """Extract and normalize numeric values from text."""
    
    # Common unit patterns
    UNITS = {
        'length': ['mm', 'cm', 'm', 'km', 'inch', 'ft', 'foot', 'feet', 'yard', 'mile'],
        'weight': ['mg', 'g', 'kg', 'lb', 'pound', 'oz', 'ounce', 'ton'],
        'volume': ['ml', 'l', 'liter', 'gal', 'gallon', 'cup', 'pint', 'quart'],
        'time': ['ms', 'sec', 'second', 'min', 'minute', 'hr', 'hour', 'day', 'week', 'month', 'year'],
        'currency': ['$', '€', '£', '¥', 'USD', 'EUR', 'GBP', 'JPY'],
        'percent': ['%', 'percent', 'percentage'],
        'temperature': ['°C', '°F', 'celsius', 'fahrenheit', 'kelvin'],
        'data': ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'byte', 'kilobyte', 'megabyte', 'gigabyte'],
    }
    
    def __init__(self, config: Config):
        """Initialize numeric extractor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract(self, text: str) -> List[NumericValue]:
        """
        Extract numeric values with units from text.
        
        Args:
            text: Input text
            
        Returns:
            List of NumericValue objects
        """
        values: List[NumericValue] = []
        
        # Pattern 1: Number with unit (e.g., "25 kg", "$100", "50%")
        pattern_with_unit = r'([$€£¥]?\s*-?\d+(?:,\d{3})*(?:\.\d+)?)\s*([a-zA-Z%°]+)?'
        
        for match in re.finditer(pattern_with_unit, text):
            num_str = match.group(1)
            unit_str = match.group(2) if match.group(2) else None
            
            # Parse number
            parsed_value = self._parse_number(num_str)
            if parsed_value is None:
                continue
            
            # Normalize unit
            normalized_unit = self._normalize_unit(unit_str) if unit_str else None
            
            values.append(NumericValue(
                value=parsed_value,
                unit=normalized_unit,
                raw_text=match.group(0).strip()
            ))
        
        # Pattern 2: Dates (YYYY-MM-DD, MM/DD/YYYY, etc.)
        date_patterns = [
            r'\b(\d{4})-(\d{2})-(\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b',  # DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                # Store as timestamp-like value
                values.append(NumericValue(
                    value=0.0,  # Placeholder; could convert to timestamp
                    unit='date',
                    raw_text=match.group(0)
                ))
        
        return values
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """Parse a number string to float."""
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$€£¥,\s]', '', num_str)
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit strings."""
        if not unit:
            return ""
        
        unit_lower = unit.lower().strip()
        
        # Check against known units
        for category, unit_list in self.UNITS.items():
            for known_unit in unit_list:
                if unit_lower == known_unit.lower():
                    return known_unit
        
        # Return as-is if not found
        return unit
