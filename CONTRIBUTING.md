# Contributing to MDKeyChunker

Thank you for considering contributing to MDKeyChunker! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/yourusername/MDKeyChunker.git
cd MDKeyChunker
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Run tests to verify setup**

```bash
pytest
```

## Code Standards

### Style Guidelines

- **PEP 8**: Follow Python style guide
- **Type hints**: All functions must have type annotations
- **Docstrings**: Use Google-style docstrings
- **Line length**: 100 characters (configured in pyproject.toml)

### Tools

We use automated tools for code quality:

```bash
# Format code
black mdkeychunker/

# Lint
ruff check mdkeychunker/

# Type checking
mypy mdkeychunker/
```

### Example Function

```python
def process_chunks(text: str, config: Config) -> list[Chunk]:
    """
    Process markdown text into enriched chunks.

    Args:
        text: Input markdown text to process
        config: Configuration object

    Returns:
        List of enriched Chunk objects

    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")

    pipeline = Pipeline(config)
    return pipeline.process_text(text)
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

### Test Structure

```python
import pytest
from mdkeychunker.module import Function

@pytest.fixture
def config():
    """Create test configuration."""
    return Config(param1=value1)

def test_basic_functionality(config):
    """Test basic behavior."""
    result = Function(config)
    assert result == expected

def test_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        Function(invalid_input)
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_chunker.py

# With coverage
pytest --cov=mdkeychunker --cov-report=html

# Verbose output
pytest -v
```

## Pull Request Process

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Verify tests pass**

```bash
pytest
mypy mdkeychunker/
black mdkeychunker/ --check
```

4. **Commit with clear messages**

```bash
git commit -m "Add feature: description of what was added"
```

5. **Push and create PR**

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- Clear title describing the change
- Description of what was changed and why
- Link to any related issues
- Screenshots/examples if applicable

## Areas for Contribution

### High Priority

- **Performance optimization**: Profile and optimize bottlenecks for large documents
- **Additional LLM providers**: Add support for Cohere, Mistral, etc.
- **Documentation**: Improve examples, tutorials, API docs
- **Streaming support**: Process very large documents chunk-by-chunk

### Medium Priority

- **Adaptive chunking**: Dynamic chunk sizing based on content
- **Streaming API**: Process large documents in chunks
- **Export formats**: Neo4j graph export, custom formats
- **Visualization**: Generate chunk relationship graphs

### Nice to Have

- **Web UI**: Simple interface for testing
- **Benchmarking suite**: Standardized performance tests
- **Integration examples**: Sample RAG pipeline implementations
- **Docker support**: Containerized deployment

## Documentation

When adding new features:

1. **Update README.md** if user-facing
2. **Add docstrings** to all functions/classes
3. **Update .env.sample** for new config options
4. **Add examples** in `examples.py` if applicable

## Code Review Process

All submissions require review. We look for:

- **Functionality**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Code quality**: Is it readable and maintainable?
- **Documentation**: Is it well-documented?
- **Performance**: Any performance implications?

## Reporting Bugs

Use GitHub Issues with:

- **Clear title**: Concise description of the bug
- **Environment**: OS, Python version, package version
- **Reproduction steps**: Minimal code to reproduce
- **Expected vs. actual behavior**
- **Logs/errors**: Full error messages

## Feature Requests

Use GitHub Issues with:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Impact**: Who would benefit?

## Questions?

- Open a GitHub Discussion for general questions
- Check existing issues/PRs for similar topics
- Tag maintainers for urgent items

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
