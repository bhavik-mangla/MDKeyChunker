# What does one LLM call per chunk buy?

Evaluation harness for the MDKeyChunker v3 study: structure-aware Markdown chunking
compared with fixed windows, a free title-chain prefix, contextual retrieval, and
single-call LLM enrichment (with and without rolling keys, with and without
key-based merging), on two public datasets.

- **Qasper** (Dasigi et al., 2021): 30 dev-set papers converted to Markdown, 79
  questions with annotator-marked evidence paragraphs. Retrieval within each paper.
- **FreshStack Laravel** (Thakur et al., 2025): 24 native Markdown files from
  `laravel/docs` at commit `1e8496c`, 73 Stack Overflow questions with
  LLM-labelled gold spans. Retrieval over the pooled corpus.

The analysis plan, written and committed before any full-run result was
computed (amendments also before results), is in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md);
its git history in this directory is the timing record.
Findings: [`results/QASPER_FINDINGS.md`](results/QASPER_FINDINGS.md),
[`results/FRESHSTACK_FINDINGS.md`](results/FRESHSTACK_FINDINGS.md).

## Setup

```bash
pip install -e ".[dev]"            # from the repository root
pip install numpy rank_bm25 requests pyarrow
ollama pull qwen2.5:7b && ollama pull mxbai-embed-large && ollama pull nomic-embed-text
cd benchmarks/chunking_study
```

Download the Qasper dev split from the Hugging Face mirror of the official
release to `data/qasper-dev.parquet`:
`https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/validation/0.parquet`.

## Run

```bash
python prepare.py --papers 30 --seed 13          # data/subset.json
python prepare_freshstack.py                     # data/freshstack_laravel.json (+ raw cache)
python orchestrate.py                            # build -> evaluate, both datasets
python validate.py data/subset.json              # content checks on built caches
python analyze.py --dataset qasper               # chunk, key-reuse and cost statistics
python analyze.py --dataset freshstack
```

Per-question values for every retriever, system and metric are in
`results/*_per_question.json` (`values["retriever|system|metric"][cluster]`, in the
order of `question_ids[cluster]`).

`orchestrate.py` expects an Ollama server on `127.0.0.1:11435` started with
`OLLAMA_NUM_PARALLEL=4` (it restarts one if needed). Every call in a run must use
the same `num_ctx` (16,384 for Qasper, 32,768 for FreshStack): Ollama reloads the
model whenever it changes, which starves concurrent requests.

## Reproducibility notes

- Results in `results/` were produced with the chunker at commit `c027b40`
  (released as 0.3.0), `qwen2.5:7b` Q4_K_M on Ollama 0.32.5, temperature 0, on an
  Apple M5 with 24 GB.
- Enrichment and contextual-retrieval calls made before the ablation rebuild did
  not pass a seed; later calls use seed 0. Rebuilds can differ slightly.
- Wall-clock times were measured under 4-way concurrency; use the token counts
  in `results/*_analysis.json` as the cost measure.
- `build.py` caches every chunk set per document under `data/cache/`, so any
  interrupted run resumes.
