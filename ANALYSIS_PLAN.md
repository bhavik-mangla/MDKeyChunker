# Analysis plan (fixed 2026-10-02 15:55 IST, before any full-run results were viewed)

Only a 2-paper smoke test (6 questions) has been run; it was used to check the
code, not to choose metrics.

## Primary metric
- Qasper: evidence recall within a 512-token budget (`rec@512t`), per question,
  averaged; 95% CI by bootstrap over papers (2,000 resamples).
- FreshStack: gold-token precision within a 1,024-token budget (`prec@1024t`),
  bootstrap over questions.
- Budgeted metrics are primary because they do not reward larger chunks.
  Hit@k and MRR are reported as secondary.

## Primary retrievers
Hybrid (BM25 + nomic, RRF) and BM25. Dense-only results are secondary:
mxbai truncates at 512 tokens, which penalises longer chunks independently of
chunking quality.

## Pre-specified comparisons (each as a paired difference with 95% CI)
1. RQ1 (does structure help?): struct/text vs fixedtok/text and fixed512/text.
2. RQ2 (does one LLM call per chunk beat free structure?):
   enr_rk/meta vs struct/tc (title-chain prefix, zero LLM calls).
3. RQ3 (enrichment vs contextual retrieval at matched call count):
   enr_rk/meta vs cr/cr.
4. RQ4 (do rolling keys matter?):
   (a) mechanism: key-reuse rate and chunks removed by merging, enr_rk vs enr_nork;
   (b) retrieval: merged_rk/meta vs merged_nork/meta, and merged_rk/meta vs enr_rk/meta.

A comparison is called a difference only if its 95% CI excludes 0 for the
primary metric on the primary retrievers. With 4 primary comparisons x 2
retrievers x 2 datasets we expect ~0.8 false positives at 95%; we report all
cells, not only significant ones, and say so.

## Cost
Report LLM calls per chunk and input/output tokens per call from Ollama
counters. Wall-clock time is reported only with the caveat that it was
measured under 4-way concurrency with a shared GPU.

## Things we will not do
No change of primary metric, budget, retriever, or question filter after
seeing results. Any post-hoc analysis is labelled exploratory.
