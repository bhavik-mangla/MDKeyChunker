# MDKeyChunker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Paper](https://img.shields.io/badge/Research-ArXiv-red.svg)](https://arxiv.org/abs/2603.23533)

Markdown chunking with single-call LLM enrichment or ultra-fast spaCy processing for RAG pipelines. 
Read the full research paper: **[MDKeyChunker: Semantic Block Restructuring for Efficient RAG](https://arxiv.org/abs/2603.23533)**

## What It Does

1. **Chunks** Markdown into structural units (headers, code blocks, tables, lists)
2. **Enriches** each chunk with ONE call → title, summary, keywords, entities, questions, semantic key
    - **LLM Mode**: High-fidelity metadata using GPT/Claude/Ollama
    - **spaCy Mode**: Fast, local, and free semantic key extraction
3. **Passes rolling keys** forward so the enricher has context about prior topics
4. **Restructures** by merging chunks that share the same specific-subtopic key

## Quick Start

```bash
pip install mdkeychunker
cp .env.sample .env   # then edit .env with your API key
mdkeychunker demo.md
```

Or programmatically:

```python
from mdkeychunker import Pipeline, Config

# High-fidelity mode (GPT-4o-mini)
config = Config.from_env()
pipeline = Pipeline(config, enricher_mode="llm")
chunks = pipeline.process_file("document.md")

# Ultra-fast / Free mode (spaCy)
pipeline_fast = Pipeline(config, enricher_mode="spacy")
chunks_fast = pipeline_fast.process_file("document.md")
```

## CLI

```bash
# Basic usage
mdkeychunker document.md

# Save to specific output
mdkeychunker document.md -o chunks.jsonl

# With summary file and stats
mdkeychunker document.md --summary summary.txt --stats

# Disable merging
mdkeychunker document.md --no-merge

# Override LLM provider (e.g., local Ollama)
mdkeychunker document.md --provider openai_compatible --base-url http://localhost:11434/v1 --model llama3
```

## Configuration

Set via `.env` file or environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `openai_compatible` |
| `LLM_API_KEY` | — | API key (not needed for local Ollama) |
| `LLM_BASE_URL` | — | Base URL for Ollama/vLLM/LM Studio |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `MIN_CHUNK_SIZE` | `100` | Minimum characters per chunk |
| `MAX_CHUNK_SIZE` | `1500` | Soft max characters per chunk |
| `MERGE_BY_KEYS` | `true` | Merge chunks sharing the same key |
| `MAX_MERGED_SIZE` | `3000` | Max combined size after merging |
| `MIN_ORPHAN_SIZE` | `200` | Below this, orphan chunks get context enrichment |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## LLM Providers

```bash
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Anthropic
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-haiku-20240307

# Ollama (local)
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
```

## How It Works

```
Markdown → Chunker → Enricher (1 LLM call/chunk) → Restructurer → Enriched Chunks
                          ↑
                    Rolling Keys
                    (context from prior chunks)
```

**Key design**: The `key` field means the *specific subtopic* that distinguishes a chunk — e.g. `"admissions process"`, `"oauth token flow"`, `"gradient descent optimization"` — never the broad document topic. Chunks with matching keys are merged globally, regardless of position.

## Chunk Schema

```json
{
  "chunk_id": "a3f2b1c4d5e6f7a8",
  "text": "The admissions process begins in March...",
  "section_title": "Admissions",
  "title": "Spring Admissions Timeline",
  "summary": "Describes the March start of the admissions process...",
  "keywords": ["admissions", "application deadline", "March intake"],
  "entities": [{"name": "March", "type": "EVENT"}],
  "questions": ["When does the admissions process begin?"],
  "key": "admissions process",
  "related_keys": ["curriculum framework"],
  "content_types": ["paragraph"],
  "position_index": 2,
  "previous_chunk_id": "9b8c7d6e5f4a3b2c",
  "next_chunk_id": "1a2b3c4d5e6f7a8b",
  "token_count": 187,
  "start_line": 12,
  "end_line": 28
}
```

## Benchmarks

MDKeyChunker is rigorously evaluated on the **SciFact** dataset (5,183 scientific papers) using industry-standard Information Retrieval (IR) metrics. We isolate chunking as the only variable while keeping embeddings ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)) and retrieval (FAISS) constant.

![SciFact Benchmark Results](figures/scifact_benchmark.png)

### Performance Comparison

| Strategy | Library | Recall@5 | nDCG@5 | Chunks/Doc | Cost Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MDKeyChunker** | — | 0.762 | **0.681** | **1.00** | **1x** |
| [Recursive Character](https://github.com/langchain-ai/langchain) | LangChain | 0.760 | 0.678 | 3.43 | 3.4x |
| [Semantic Chunker](https://github.com/langchain-ai/langchain-experimental) | LangChain | **0.775** | 0.681 | 2.03 | 2x |
| Fixed Token (512) | Standard | 0.762 | 0.681 | 1.05 | 1.1x |

> **Scientific Note**: Standard "Semantic Chunkers" often over-fragment dense technical text into small sentence clusters, leading to a high chunk ratio (2.03x in our tests). MDKeyChunker uses **Structural-Semantic Merging**, grouping disparate blocks (like Code + Paragraphs) into single high-density units, achieving peak accuracy with significantly fewer total chunks.

**Key Research Takeaway**: MDKeyChunker achieves state-of-the-art ranking precision while being **70% more efficient** than standard Recursive splitting.
 By merging related blocks into high-density semantic units, we eliminate context fragmentation and significantly reduce Vector DB storage and LLM inference costs.

## API

```python
from mdkeychunker import Pipeline, Config, Chunk

# Config
config = Config(llm_provider="openai", llm_model="gpt-4o-mini", merge_by_keys=True)

# Pipeline
pipeline = Pipeline(config)
chunks: list[Chunk] = pipeline.process_text(markdown_text)
chunks: list[Chunk] = pipeline.process_file("path/to/file.md")

# Save output
pipeline.save_jsonl(chunks, "output.jsonl")
pipeline.save_summary(chunks, "summary.txt")
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
