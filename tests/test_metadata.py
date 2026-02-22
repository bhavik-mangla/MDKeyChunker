"""Tests for metadata extractors."""

import pytest
from mdkeychunker.metadata.entity_extractor import EntityExtractor
from mdkeychunker.metadata.keyword_extractor import KeywordExtractor
from mdkeychunker.metadata.numeric_extractor import NumericExtractor
from mdkeychunker.metadata.text_cleaner import TextCleaner
from mdkeychunker.utils.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        entity_extraction_mode="none",  # Skip spaCy for basic tests
        strict_entity_mode=True
    )


def test_keyword_extraction(config):
    """Test keyword extraction."""
    extractor = KeywordExtractor(config)
    
    text = """
    Machine learning models use neural networks to process data.
    Deep learning is a subset of machine learning that uses multiple layers.
    """
    
    keywords = extractor.extract(text)
    
    assert len(keywords) > 0
    # Should extract meaningful terms
    assert any('learning' in kw.lower() for kw in keywords)


def test_numeric_extraction(config):
    """Test numeric value extraction."""
    extractor = NumericExtractor(config)
    
    text = "The model achieved 95.5% accuracy on the dataset of 10,000 samples."
    
    values = extractor.extract(text)
    
    assert len(values) > 0
    
    # Should find percentage
    percent_values = [v for v in values if v.unit and '%' in v.unit]
    assert len(percent_values) > 0
    
    # Should find the number
    number_values = [v for v in values if v.value == 10000.0]
    assert len(number_values) > 0


def test_numeric_extraction_with_units(config):
    """Test extraction of numbers with various units."""
    extractor = NumericExtractor(config)
    
    text = "The server has 256 GB RAM, 2 TB storage, and runs at 3.5 GHz."
    
    values = extractor.extract(text)
    
    assert len(values) >= 3
    
    # Check some values
    gb_values = [v for v in values if v.unit and 'GB' in v.unit]
    assert len(gb_values) > 0


def test_text_cleaning(config):
    """Test text cleaning."""
    cleaner = TextCleaner(config)
    
    text = "This  has   extra    spaces and \"smart quotes\" and—dashes."
    
    cleaned, char_map = cleaner.clean(text)
    
    # Should normalize whitespace
    assert '   ' not in cleaned
    
    # Should normalize quotes
    assert '"smart quotes"' in cleaned or "'smart quotes'" in cleaned
    
    # Should normalize dashes
    assert '—' not in cleaned


def test_text_cleaning_preserves_structure(config):
    """Test that cleaning preserves paragraph structure."""
    cleaner = TextCleaner(config)
    
    text = """First paragraph.

Second paragraph.

Third paragraph."""
    
    cleaned, char_map = cleaner.clean(text)
    
    # Should preserve paragraph breaks
    assert '\n\n' in cleaned or cleaned.count('\n') >= 2


def test_entity_extraction_with_spacy():
    """Test entity extraction with spaCy (if available)."""
    config = Config(entity_extraction_mode="spacy", strict_entity_mode=True)
    extractor = EntityExtractor(config)
    
    if not extractor.nlp:
        pytest.skip("spaCy model not available")
    
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    cleaned = text
    
    entities = extractor.extract(text, cleaned)
    
    # Should extract entities
    assert len(entities) > 0
    
    # Should have organization or person entities
    entity_types = {e.type for e in entities}
    assert entity_types.intersection({'ORG', 'PERSON', 'GPE'})
