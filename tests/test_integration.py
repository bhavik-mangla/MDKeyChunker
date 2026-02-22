"""Integration tests for full pipeline."""

import pytest
from mdkeychunker.pipeline.processor import DocumentProcessor
from mdkeychunker.utils.config import Config


@pytest.fixture
def processor():
    """Create processor with test configuration."""
    config = Config(
        entity_extraction_mode="none",  # Skip spaCy for speed
        llm_call_budget_per_doc=0,  # No LLM calls in tests
        pass_keys_forward=True,
        duplicate_detection=True,
        min_chunk_size=50,
        soft_max_chunk_size=500,
        hard_max_chunk_size=1000
    )
    return DocumentProcessor(config)


def test_full_pipeline(processor):
    """Test complete processing pipeline."""
    markdown = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It focuses on training models using data.

## Supervised Learning

In supervised learning, we have labeled data. The model learns from examples with known outputs.

### Example: Classification

Classification assigns data points to predefined categories. Common algorithms include:
- Decision Trees
- Random Forests
- Neural Networks

## Unsupervised Learning

Unsupervised learning works with unlabeled data. The goal is to discover patterns.

```python
# Example clustering code
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(data)
```

## Results

| Algorithm | Accuracy | Speed |
|-----------|----------|-------|
| Tree      | 85%      | Fast  |
| Forest    | 92%      | Medium|
| Neural    | 95%      | Slow  |

The neural network achieved 95% accuracy, which is the best result.
"""
    
    chunks = processor.process_text(markdown)
    
    # Should produce multiple chunks
    assert len(chunks) > 0
    
    # All chunks should have metadata
    for chunk in chunks:
        assert chunk.metadata is not None
        assert chunk.metadata.chunk_id is not None
        assert chunk.metadata.position_index >= 0
    
    # Check content types
    content_types = set()
    for chunk in chunks:
        content_types.update(chunk.content_types)
    
    # Should include various types
    assert len(content_types) > 0
    
    # Should have extracted some keywords
    total_keywords = sum(len(c.metadata.keywords) for c in chunks)
    assert total_keywords > 0
    
    # Should have numeric values (from table)
    total_numerics = sum(len(c.metadata.numeric_values) for c in chunks)
    assert total_numerics > 0


def test_navigation_links(processor):
    """Test that chunks are linked with prev/next IDs."""
    markdown = """# Section 1

Content 1.

# Section 2

Content 2.

# Section 3

Content 3.
"""
    
    chunks = processor.process_text(markdown)
    
    if len(chunks) > 1:
        # Check forward links
        for i in range(len(chunks) - 1):
            assert chunks[i].metadata.next_chunk_id == chunks[i+1].metadata.chunk_id
        
        # Check backward links
        for i in range(1, len(chunks)):
            assert chunks[i].metadata.previous_chunk_id == chunks[i-1].metadata.chunk_id


def test_statistics(processor):
    """Test statistics generation."""
    markdown = """# Test Document

This is a test with some content.

## Section

More content here.
"""
    
    chunks = processor.process_text(markdown)
    stats = processor.get_statistics(chunks)
    
    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(chunks)
    assert "total_entities" in stats
    assert "total_keywords" in stats
    assert "content_types" in stats


def test_empty_document(processor):
    """Test handling of empty document."""
    markdown = ""
    
    chunks = processor.process_text(markdown)
    
    # Should handle gracefully
    assert isinstance(chunks, list)


def test_markdown_only_headers(processor):
    """Test document with only headers."""
    markdown = """# Header 1
## Header 2
### Header 3
"""
    
    chunks = processor.process_text(markdown)
    
    # Should produce chunks
    assert len(chunks) > 0
