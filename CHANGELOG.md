# Changelog

## 0.4.1 — 2026-10-17

### Fixed
- `nomic` dense preset: the `[dense]` extra now installs `einops`, which the model's remote code
  (`nomic-ai/nomic-bert-2048`) imports. Before, `diagnose(..., embedder="nomic")` failed with
  `ImportError: ... einops`.
- `nomic` preset on transformers 5: its remote code crashes there
  (`'NomicBertModel' object has no attribute 'get_extended_attention_mask'`; an upstream fix is
  pending in nomic-ai/nomic-bert-2048#23). Loading it now fails at once with
  `ImportError: the 'nomic' embedder needs transformers<5 ... pip install 'transformers<5'`, for the
  preset name and for its Hugging Face id, instead of a traceback at the first encode.
  `qwen3-embedding-0.6b`, `mxbai` and both reranker presets work on transformers 5.
- Scores no longer depend on the transformers version. Transformers 5 loads a model in its
  config's dtype, so `qwen3-embedding-0.6b` ran in bfloat16 there (float32 on transformers 4)
  and its scores moved by up to 3e-3 (mxbai: 9e-5), which changed `diagnose` numbers. Dense and
  reranker presets now load in float32 by default, and give bit-identical scores on
  transformers 4.57 and 5.20 on the same machine. Pass `dtype=None` (the model's default, faster)
  or `model_kwargs=` to `DenseRetriever`, `load_embedder` or `Reranker` to override. The
  reranker's cache key is unchanged for float32, so existing caches stay valid.

- `mk.diagnose` on a large corpus could exhaust a laptop's memory: the built-in embedders
  encode in batches of 32 at up to 8,192 tokens, and the OS killed the process without a
  traceback (3,896 chunks, nomic + qwen3, 24 GB). The batch size and token cap can now be set
  (see Added).

- The `[dense]` / `[rerank]` lower bounds did not work together: the lowest allowed versions
  could not load a model (torch 2.0 lacks `torch.compiler`; transformers 4.51.0 fails with
  `NameError: init_empty_weights`; torch < 2.3 cannot hand results to numpy 2). The floors
  are now `sentence-transformers>=3.0.0`, `transformers>=4.51.1`, `torch>=2.3.0` (and
  `einops>=0.7.0` for `[dense]`), and CI's lowest-direct job loads two tiny models through
  `load_embedder`, `DenseRetriever` and `Reranker` to prove it.

### Added
- `mk.diagnose(..., embed_batch_size=, max_seq_length=)` and the CLI's `--embed-batch-size` /
  `--max-seq-length`, for the built-in embedders (defaults unchanged; e.g. 4 and 2048 on a
  laptop). `embedder=load_embedder("nomic", batch_size=4, max_seq_length=2048)` also works and
  keeps the preset's prefixes.
- `dtype=` and `model_kwargs=` on `DenseRetriever`, `load_embedder` and `Reranker`.
- Slow tests that load every dense and reranker preset and check a toy retrieval. The weekly
  CI job runs them with the default dependency resolution and with transformers 4, then checks
  that both give the same preset scores (to 1e-6), so a preset that breaks or drifts on a new
  transformers release is caught there.

## 0.4.0 — 2026-10-17

Rebuilt around the findings of the v4 study (arXiv:2603.23533 v4, *Upgrade the Retriever First:
What an LLM Call per Chunk Buys in RAG*): free structure first, a strong retriever, and LLM
calls only where they pay.

### Added
- `mk.chunk()`, `mk.chunk_file()`, `mk.window()`: header-led chunks with a token cap on what
  gets embedded, row- and line-aware splitting of oversized tables and code, a fallback to
  256-token windows for documents without Markdown structure, character offsets into the
  source, `section_path`, `embed_text` with the section-path prefix, and content-stable ids.
- `mk.diagnose()`: checks your retrieval stack on your documents and labelled questions
  (fusion vs its dense half, embedder upgrade, reranker, enrichment), with paired bootstrap
  CIs, Holm adjustment, the minimum detectable difference, and "inconclusive" or
  "identical" verdicts instead of a winner when the evidence is thin. Accepts your own
  retrievers or search functions.
- `mk.evidence_recall()` and `mk.evidence_precision()`: gold-evidence metrics within a token
  budget that work with any chunking (the study's Qasper and FreshStack metrics).
- Reference retrievers `BM25Retriever` (`method="lucene"` default, `method="study"` reproduces
  the paper), `DenseRetriever`, `HybridRetriever` with `RRF` or `Convex` fusion, and a Qwen3
  reranker; presets for qwen3-embedding, nomic, mxbai and bge.
- `mk.enrich()` / `mk.aenrich()`: one call per chunk for all fields, the study's prompt,
  concurrency across documents, an on-disk cache, fail-fast on authentication errors, one warning when
  enriched `embed_text` exceeds the cap, and `max_field_tokens` to trim the generated fields
  (never the chunk text). `mk.OpenAICompatible` and
  `mk.Anthropic` clients.
- `mk.estimate_cost()`: tokens and dollars for enrichment versus contextual retrieval, with
  optional prompt caching.
- Framework adapters: `MDKeyChunkerSplitter` (LangChain `TextSplitter`), `MDKeyChunkerNodeParser`
  (LlamaIndex `NodeParser`), and `mdkeychunker.integrations.to_chunks` to score chunks from
  LangChain, LlamaIndex, Chonkie, dicts or strings.
- `mdkeychunker chunk` and `mdkeychunker diagnose` CLI subcommands.
- `tokenizer=` accepts a tiktoken encoding name (`"cl100k_base"`) in `mk.chunk`, `mk.window`,
  `mk.enrich` and `mk.evidence_recall` alike.
- Statistics module (paired cluster bootstrap with BCa intervals, sign-flip test, TOST, Holm).
- Typed package (`py.typed`), property-based tests of the chunker's invariants, a
  differential test against markdown-it-py, golden-file tests on 11 real documents, and
  parity tests against the v4 study's data.

### Changed (breaking)
- **No required dependencies.** `openai`, `python-dotenv` and `tiktoken` moved to extras:
  `[llm]`, `[cli]`, `[tokens]`. New extras: `[bm25]`, `[dense]`, `[rerank]`, `[retrieval]`,
  `[langchain]`, `[llamaindex]`, `[all]`.
- **`merge_by_keys` defaults to off.** The study found no retrieval gain from key-based merging.
- The library no longer reads `.env` or calls `logging.basicConfig`; `Config.from_env()`
  reads a file only when you pass `env_file=`. `mdkeychunker legacy` still loads `./.env` when `[cli]` is installed.
- ATX headers indented by 1–3 spaces are recognised (CommonMark).
- `[langchain]` installs `langchain-text-splitters` (which provides `TextSplitter`).
- The CLI no longer calls an LLM unless you run `mdkeychunker legacy`; a bare `mdkeychunker FILE`
  is an alias of `mdkeychunker chunk FILE`.

### Deprecated
- `Pipeline` and `min_orphan_size` (removal planned for 0.6). Use `mk.chunk()` and `mk.enrich()`.
- `Config.from_env()` reading `./.env` without `env_file=` (now warns and skips the file).

## 0.3.0 — 2026-10-02

### Fixed
- **LLM enrichment works again.** `ENRICH_PROMPT` had been replaced by a truncated stub in 0.2.0's June commits, so the chunk text never reached the model and LLM mode returned no keys or summaries.
- One failed LLM call no longer aborts the document. Authentication errors and unknown providers fail fast.
- Malformed LLM output is coerced to the expected types. `related_keys` are lowercased and limited to keys from the rolling dictionary.
- **Chunker:** headers now start a chunk and always stay with their content (previously about a third of chunks on header-dense documents ended on a dangling header). Small chunks are merged forward, and the larger part names the section.
- **Chunker:** code fences follow CommonMark (fence length and character, at most 3 spaces of indent, no backticks in a backtick info string). Setext headers, CRLF line endings, `1)` lists and trailing `#`s are handled.

### Changed
- Chunk boundaries differ from 0.2.x on most documents. Re-index after upgrading.
- `spacy` and `anthropic` are optional extras (`pip install "mdkeychunker[spacy]"`, `"mdkeychunker[anthropic]"`).
- The SciFact benchmark moved out of the installed package into `benchmarks/`.

### Added
- `CITATION.cff`, a citation section in the README, and CI on Python 3.10–3.13.

## 0.2.0
- Single-call enrichment with rolling keys and key-based restructuring (the version described in arXiv:2603.23533 v1/v2, commit `e3e1b86`).
