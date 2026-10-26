# MDKeyChunker

[![PyPI](https://img.shields.io/pypi/v/mdkeychunker.svg)](https://pypi.org/project/mdkeychunker/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2603.23533-b31b1b.svg)](https://arxiv.org/abs/2603.23533)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-mdkeychunker--study-yellow)](https://huggingface.co/datasets/bhavikmangla/mdkeychunker-study)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhavik-mangla/MDKeyChunker/blob/main/examples/diagnose_quickstart.ipynb)

**Header-aware chunking for RAG, and a diagnostic that tells you where to spend:
on the retriever, on chunking, or on LLM calls per chunk.**

Every default in this package comes from a confirmatory study of 951 Qasper and 235
FreshStack questions under eight retrievers:
*[Upgrade the Retriever First: What an LLM Call per Chunk Buys in RAG](https://arxiv.org/abs/2603.23533)*.

```bash
pip install mdkeychunker          # zero dependencies
```

```python
import mdkeychunker as mk

chunks = mk.chunk(markdown, doc_id="guide")   # header-led chunks, <= 512 tokens each
chunks[0].embed_text                            # "Guide > Install > Linux\n\n..." - embed / index this
```

No LLM call, no API key, no model download. Every chunk carries its section path,
character offsets into the source, a content-stable id and a token count.

## Check your retrieval stack first

`mk.diagnose` runs on your own documents and a few labelled questions. It tells you
whether your hybrid search beats its own dense half, whether a stronger embedder or a
reranker would help, and whether LLM enrichment would pay off for your retriever, with
paired bootstrap confidence intervals and an honest "inconclusive" when you have too few
questions.

```python
qa = [
    {"question": "How do I rotate the API token?", "doc_id": "guide",
     "gold": ["Rotate the token from Settings > API"]},    # text that answers it
    # ... 30-100 questions is a good start
]
report = mk.diagnose(chunks, qa)          # pip install "mdkeychunker[retrieval]"
print(report.to_markdown())
```

Each comparison ends in a plain verdict. On the study's own Qasper data (951 questions,
300 papers), comparing BM25 over enriched chunks with BM25 over the free prefix, it prints:

```
bm25@enriched beats bm25. +8.7 points [+6.6, +10.8]; MDE ~3.0 points.
```

That is the paper's result, reproduced to the last digit. With too few questions you get
"Inconclusive at n=..." and the smallest difference your questions could detect, instead of
a winner. If your token budget covers whole documents, so that every retriever finds the
evidence, it says so.

On a laptop with a large corpus, lower the built-in embedders' memory use with
`mk.diagnose(..., embed_batch_size=4, max_seq_length=2048)` (CLI: `--embed-batch-size 4
--max-seq-length 2048`). The defaults (batch 32, up to 8,192 tokens per chunk for qwen3 and
nomic) can exhaust memory, and the OS then kills the process without a traceback.

Bring your own retriever as a function `search(query, k) -> hits | chunks | chunk ids`,
or pass LangChain / LlamaIndex outputs. `diagnose` scores gold *evidence text*, not chunk
ids, so you can compare different chunkers on the same questions.

Try it in one click: the [quickstart notebook](examples/diagnose_quickstart.ipynb) chunks a document, runs
`diagnose` on 10 real research papers and compares two chunkers on the same questions.

## Command line

```bash
mdkeychunker chunk docs/guide.md -o chunks.jsonl            # header-led chunks with the free prefix
mdkeychunker chunk notes.txt --windows --size 256           # token windows
mdkeychunker diagnose --docs docs/ --qa qa.jsonl            # is your retriever leaving recall on the table?
```

`qa.jsonl` holds one question per line:
`{"question": "...", "gold": ["evidence text", ...], "doc_id": "guide.md"}`. With `--docs DIR`,
a document's `doc_id` is its path relative to `DIR`. No command calls an LLM unless you run
`mdkeychunker legacy` (the 0.3 pipeline).

## What to use when

From the v4 study. Each row gives the evidence tier: **confirmatory** (pre-planned
primary test), **secondary** (arm or retriever fixed before any outcome), **exploratory**
(analysed after results were seen). Scope: English, Qasper (search within a paper) and
FreshStack (code documentation), 7B and 120B indexing LLMs.

| Your situation | Do | Evidence |
|---|---|---|
| Any Markdown corpus, no budget | `mk.chunk()` with the section-path prefix (default) | Header-led chunks beat 512-character windows on both datasets (confirmatory). The prefix helps some embedders (mxbai +2.9 on Qasper, exploratory) |
| Code or API documentation | Also try `mk.window(text, size=256, overlap=32)` | 256-token windows beat header chunks on FreshStack by 2.9 points (confirmatory) |
| Hybrid BM25 + dense with RRF | Run `mk.diagnose()`; if hybrid < dense, tune the fusion or drop BM25 | Untuned RRF (k=60) scored below its own dense half on Qasper: 42.7 vs 47.8 (exploratory) |
| You have budget to spend | Upgrade the embedder or add a reranker **before** paying for LLM indexing | With the free prefix, qwen3-embedding or a reranker beat hybrid enrichment by 3.4 to 6.3 points (exploratory) |
| Strong retriever already, 7B LLM | Skip enrichment for retrieval | Under qwen3-embedding and a Qwen3 reranker, 7B enrichment is equivalent to the free prefix within the study's margins on both datasets (secondary). A 120B LLM kept a +2.5 edge at the top of the Qasper grid (exploratory) |
| BM25 or keyword-led search | Enrichment works as document expansion: `mk.enrich()` | +8.7 points under BM25 on Qasper (secondary) |
| Considering contextual retrieval | Use one-call enrichment instead; compare costs with `mk.estimate_cost()` | Equivalent retrieval (confirmatory) with 5.8 to 8.7 times fewer input tokens per call on these documents; the gap grows with document length |
| One LLM call per field | Use one call for all fields | Equivalent (secondary) |
| Rolling key dictionaries, key-based merging, related-key links | Skip them for retrieval | No gain over matched LLM-free controls; related-key expansion lowered Qasper recall by 10.0 points (secondary) |
| Late chunking to search within one long document | Skip it | Lowered within-paper recall by 11.4 points (secondary) |

## Install

The core has no dependencies. Add extras for what you use:

| Extra | Gives you |
|---|---|
| `mdkeychunker[llm]` | `mk.enrich` with OpenAI-compatible servers (OpenAI, Ollama, vLLM, LM Studio) |
| `mdkeychunker[anthropic]` | `mk.enrich` with Claude |
| `mdkeychunker[tokens]` | exact token counts with tiktoken (default: a characters/4 estimate) |
| `mdkeychunker[bm25]` | fast BM25 (bm25s); `BM25Retriever(method="study")` needs nothing |
| `mdkeychunker[dense]`, `[rerank]` | sentence-transformers: qwen3-embedding, nomic, mxbai; Qwen3 and bge rerankers |
| `mdkeychunker[retrieval]` | all three above, for `mk.diagnose` with built-in retrievers |
| `mdkeychunker[langchain]`, `[llamaindex]` | framework adapters (`langchain-text-splitters`, `llama-index-core`) |
| `mdkeychunker[cli]` | `.env` support for `mdkeychunker legacy` and `Config.from_env` |
| `mdkeychunker[spacy]` | the legacy spaCy enricher (experimental, not covered by the study) |
| `mdkeychunker[all]` | everything |

## Chunking

```python
chunks = mk.chunk(markdown, doc_id="guide",
                  max_tokens=512,    # caps embed_text (prefix included); None = no cap
                  prefix=True,       # prepend the section path to embed_text
                  tokenizer=None)    # None = chars/4 estimate; "cl100k_base" or any tiktoken encoding name (needs [tokens])
chunks = mk.chunk_file("docs/guide.md")       # .md / .txt; convert PDFs first (see below)
windows = mk.window(text, size=256, overlap=32)
```

- Chunks follow headers. A chunk over `max_tokens` is re-packed at block boundaries, so
  code fences, tables, lists and paragraphs stay whole. Only a block that alone exceeds the
  cap is split: code by lines and tables by rows (the fence or header row is repeated in
  each piece), prose by paragraph, line, sentence, then word. Those pieces are marked
  `oversized=True`.
- Documents with no Markdown structure (no headers, or a single header and no code, tables
  or lists, as in plain text and many PDF conversions) fall back to 256-token windows with a
  `HeaderlessWarning`. Code blocks and tables are never cut by windows. Force either mode
  with `headerless="structure"` or `headerless="window"`.
- `chunk.text` is the raw text, `chunk.embed_text` is what you index,
  `chunk.section_path` is the header stack, and `chunk.start_char` / `chunk.end_char`
  point into the original input.

PDFs and Word files: convert them to Markdown first, for example with
[Docling](https://github.com/docling-project/docling), then call `mk.chunk`.

## Enrichment (optional)

One LLM call per chunk writes a title, summary, keywords and likely questions. `embed_text`
becomes those fields, a blank line, then the chunk text: the layout the study indexed, which
does not include the section path (`prefix=True` puts it back first). The fields are also
stored on the chunk (`chunk.title`, `chunk.keywords`, ...). Use it when BM25 drives your
ranking, or when you want the metadata itself.

```python
llm = mk.OpenAICompatible(model="qwen2.5:7b", base_url="http://localhost:11434/v1")  # or mk.Anthropic(...)
enriched = mk.enrich(chunks, llm, cache_dir=".mk_cache")   # one call per chunk, all fields; reruns are free
# prices in dollars per million tokens; on short documents contextual retrieval can be the cheaper one
print(mk.estimate_cost(chunks, method="enrich", price_in=0.15, price_out=0.60))
print(mk.estimate_cost(chunks, method="contextual", price_in=0.15, price_out=0.60))
```

The prompt is the one the study used. Calls are concurrent across documents, cached on
disk, and stop at the first authentication error. If generated fields push `embed_text`
over `max_tokens` (pass the value you used in `mk.chunk`; default 512), you get one warning; `max_field_tokens=` trims the fields, never the
chunk text. `rolling_keys=True` shows the model the keys used so far in the document; it
makes keys more consistent, but did not change retrieval in the study.
`estimate_cost(..., cached=True)` models prompt caching for contextual retrieval.

## LangChain and LlamaIndex

```python
from mdkeychunker.integrations.langchain import MDKeyChunkerSplitter      # pip install "mdkeychunker[langchain]"
docs = MDKeyChunkerSplitter(max_tokens=512).split_documents(loaded_docs)
# page_content = embed_text; metadata: section, start_index, end_index, chunk_id, ...

from mdkeychunker.integrations.llamaindex import MDKeyChunkerNodeParser   # pip install "mdkeychunker[llamaindex]"
nodes = MDKeyChunkerNodeParser(max_tokens=512).get_nodes_from_documents(documents)
# node ids = chunk ids; the embedded text is exactly mk.chunk's embed_text
```

Both adapters hand the framework `embed_text` by default. Pass `content="text"` for the
raw chunk text, with the section path in `metadata["section"]`. To score chunks from
another library with `mk.diagnose` or `mk.evidence_recall`, convert them with
`mdkeychunker.integrations.to_chunks(...)`. It accepts LangChain documents, LlamaIndex
nodes, Chonkie chunks, dicts or strings, and needs no extra.

## Retrievers and metrics

Reference retrievers for evaluation, matching the study's setup:

```python
r = mk.HybridRetriever(chunks, dense="qwen3-embedding-0.6b",
                       fusion=mk.RRF(k=60),             # or mk.Convex(alpha=0.5)
                       rerank="qwen3-reranker-0.6b")    # reorders the top 30
hits = r.search("how do I rotate the token?", k=10)

mk.evidence_recall(hits, gold=["Rotate the token from Settings > API"], budget_tokens=512)
```

- `BM25Retriever(method="lucene")`, the default, uses the Elasticsearch/OpenSearch formula.
  `method="study"` reproduces the paper's BM25 exactly.
- `evidence_recall` is the share of gold evidence found within a token budget (the study's
  Qasper metric). `evidence_precision` is the share of retrieved text that is gold (its
  FreshStack metric). Both work with any chunking.

For production search, use your vector database's hybrid search and set it from what
`diagnose` tells you.

## Upgrading from 0.3

- **The core has no dependencies.** LLM mode needs `pip install "mdkeychunker[llm]"`.
- **`merge_by_keys` is off by default.** The study found no retrieval gain from key-based merging.
- **`Pipeline` is deprecated** (removal in 0.6) and still works. Use `mk.chunk` plus `mk.enrich`.
- **The library no longer reads `.env` or configures logging.** Call
  `Config.from_env(env_file=".env")` (needs `pip install "mdkeychunker[cli]"`) if you relied
  on it. `mdkeychunker legacy` still reads `./.env` when `[cli]` is installed.
- **`mk.chunk` gives content-stable ids** (`chunk_id` = hash of doc_id, start offset and
  text). They don't change when enrichment runs or a document grows after the chunk.
  `Pipeline` keeps the 0.3 ids.
- **`mk.chunk` caps chunks at 512 tokens of `embed_text`** (prefix included); `Pipeline`
  keeps the 0.3 character sizes.
- **The CLI no longer calls an LLM** unless you run `mdkeychunker legacy`.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## The study

The harness, analysis plans with their full git history, and per-question results are in
[`benchmarks/chunking_study/`](benchmarks/chunking_study/). The generated enrichment and
contextual-retrieval texts are on
[Hugging Face](https://huggingface.co/datasets/bhavikmangla/mdkeychunker-study), so every
retrieval result can be re-scored without an LLM call. This package's metrics, statistics
and BM25 reproduce the study's numbers; tests check that against the study data.

`mk.chunk(..., max_tokens=None)` reproduces the study's header-led chunks exactly (all 677
cached documents, 14,274 chunks). The default `embed_text` prefix is the header path at the
chunk's first line. The study's prefix used `chunk.section_title`, which for a small chunk
merged across a header names the later section. The two differ on 2.3% of chunks;
`chunk.section_title + "\n\n" + chunk.text` is the study's exact text.

## Citation

```bibtex
@article{mangla2026upgrade,
  title   = {Upgrade the Retriever First: What an {LLM} Call per Chunk Buys in {RAG}},
  author  = {Mangla, Bhavik},
  journal = {arXiv preprint arXiv:2603.23533},
  year    = {2026}
}
```

## License

MIT. Test documents keep their own licences; see `tests/golden/SOURCES.md`.
