# Upgrade the Retriever First: study harness

Evaluation harness, analysis plans and per-question results for the paper
*Upgrade the Retriever First: What an LLM Call per Chunk Buys in RAG*
([arXiv:2603.23533](https://arxiv.org/abs/2603.23533), v4).

The study asks what one LLM call per chunk at indexing time buys over free document
structure and over a better retriever. It compares fixed windows, header-led chunks,
a free section-path prefix (S+T), single-call enrichment (E), contextual retrieval (CR),
their 120B-indexed versions, and the add-ons built on the call (rolling key dictionary,
key-based merging, related-key expansion) against matched LLM-free controls, under
eight retrievers.

| Dataset | Questions | Corpus | Retrieval | Primary metric |
|---|---|---|---|---|
| Qasper (Dasigi et al., 2021) | 951 | 300 papers, converted to Markdown | within each paper | evidence recall within 512 tokens |
| FreshStack (Thakur et al., 2025) | 235 (Laravel 89, Angular 99, YOLO 47) | three documentation repositories | pooled per domain | gold-token precision within 1,024 tokens |

## Plans and timing record

- [`ANALYSIS_PLAN_v2.md`](ANALYSIS_PLAN_v2.md): the confirmatory plan, pushed to GitHub
  before the first index was built. Hypotheses, metrics, primary retriever (BM25 + nomic,
  RRF k=60), equivalence margins (±5 Qasper, ±4 FreshStack) and Holm families.
- [`ANALYSIS_PLAN_v2_AMENDMENT_1.md`](ANALYSIS_PLAN_v2_AMENDMENT_1.md) (before any outcome:
  reranker, BM25+qwen3, late chunking, FreshStack answers, scaling arm) and Amendments
  [2](ANALYSIS_PLAN_v2_AMENDMENT_2.md)–[4](ANALYSIS_PLAN_v2_AMENDMENT_4.md) (add-on tests,
  written after H1–H3 were viewed and pushed before their arms were built; secondary).
- [`DEVIATIONS.md`](DEVIATIONS.md): every deviation from the plan, dated.
- The git history of this directory is the timing record. The pilot plan
  [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) is exploratory (see "Pilot" below).

## Results

`results/v2/` holds everything the paper reports:

- `stats.json`, `summary.txt`: confirmatory and key-secondary tests (H1–H3, 120B arms, Qasper answer F1).
- `h4_*`, `h5_*`, `dial_*`: add-on tests (Amendments 2–4).
- `extra/`, `extra_qwen3/`: secondary retrievers (reranker, qwen3-embedding, BM25+qwen3, jina-v3, late chunking).
- `qasper_e2e*`: Qasper answer-level outcomes.
- `analysis_v4.json`: exploratory analyses (document-expansion mechanism, retriever frontier, heterogeneity, budgets).
- `*_per_question.json`: per-question values for every arm, retriever and metric
  (`values["retriever|system|metric"][cluster]`, in the order of `question_ids[cluster]`).
- `frozen_config.json`: models, digests, decoding settings and prompt hashes fixed before the first build.

## Reproduce the paper's numbers (no models, no API calls)

```bash
cd benchmarks/chunking_study
python paper_numbers.py --out /tmp/paper/numbers.tex   # also writes grids.tex and the three TikZ figures
```

`paper_numbers.py` reads only `results/` and regenerates every number, grid and figure in
the paper. The per-arm token and cost figures also need the generation caches (next section).

## Re-score retrieval without LLM calls

The generated enrichment and contextual-retrieval texts for every chunk (about 535 MB) are on
Hugging Face: [`bhavikmangla/mdkeychunker-study`](https://huggingface.co/datasets/bhavikmangla/mdkeychunker-study).

```bash
hf download bhavikmangla/mdkeychunker-study --repo-type dataset --local-dir data/v2
python evaluate_v2.py          # BM25, nomic, hybrid (primary), mxbai
```

With the caches in `data/v2/cache/`, every retriever and statistic can be recomputed
without a single LLM call. Embeddings need Ollama with `nomic-embed-text`,
`mxbai-embed-large` and `qwen3-embedding:0.6b`; the secondary retrievers need a
Python environment with torch, transformers and sentence-transformers.

## Full rerun

```bash
pip install -e ".[dev]"                                   # from the repository root
pip install numpy rank_bm25 requests pyarrow tiktoken
ollama pull qwen2.5:7b && ollama pull nomic-embed-text && ollama pull mxbai-embed-large && ollama pull qwen3-embedding:0.6b

python prepare_v2_qasper.py && python prepare_v2_freshstack.py && python select_v2.py
python orchestrate_v2.py          # builds (7B local, 120B API) -> evaluate_v2 -> e2e_v2; idempotent, resumes
python build_h4.py && python evaluate_h4.py && python evaluate_h5.py     # add-on tests
python evaluate_extra.py          # secondary retrievers (ML environment)
python e2e_freshstack.py          # FreshStack answers + judge (secondary)
python analysis_v4.py             # exploratory analyses
python paper_numbers.py --out <paper>/numbers.tex
```

Data selection is fixed in `data/v2/selection.json` (see `data/v2/DATASETS_V2.md`).
`orchestrate_v2.py --estimate` prints the expected cost and time before running.
The 120B arms use `openai/gpt-oss-120b` through OpenRouter (key in `.env`, never committed);
total API spend for the study was US$3.71.

## Reproducibility notes

- Local indexing LLM: `qwen2.5:7b` Q4_K_M on Ollama, temperature 0, seed 0; one Ollama
  server on `127.0.0.1:11435` with `OLLAMA_NUM_PARALLEL=4`. Keep one `num_ctx` per run
  (Ollama reloads the model when it changes); the values are in `frozen_config.json`.
- Builds refuse to run if the configuration differs from `frozen_config.json`.
- LLM parse failures are retried with seeds 1–3, after which the chunk keeps empty fields
  and is counted.
- Every stage caches per document, so interrupted runs resume.
- The chunker evaluated is release 0.3.0 of this package.

## Pilot (exploratory)

The v3 pilot (30 Qasper papers, 79 questions; 73 FreshStack Laravel questions) used
`prepare.py`, `prepare_freshstack.py`, `orchestrate.py` and `analyze.py`, with results in
`results/*_FINDINGS.md`. Its plan was pushed only after its results existed, so it is
exploratory; it set the confirmatory study's sample sizes and margins. A wrong Laravel
commit in the pilot is corrected in the paper's appendix.
