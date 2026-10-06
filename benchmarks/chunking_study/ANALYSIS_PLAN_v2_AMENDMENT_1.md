# Amendment 1 to analysis plan v2 (made before any v2 outcome metric was viewed)

Date: 2026-10-06. At the time of this amendment the v2 builds were running and
no v2 retrieval, answer or judge metric had been computed. Only build
monitoring (cost, parse failures, throughput) had been viewed, as allowed by
plan v2 section 9. Nothing in plan v2 sections 1-5 changes: samples, primary
outcomes, primary retriever, hypotheses, margins, families and Holm
adjustments stay exactly as written.

All additions below are **secondary or exploratory**, reported with unadjusted
95% CIs and labelled as such. They respond to an internal, AI-assisted mock
review of the v3 paper and a literature check of 2025-2026 chunking evaluations.
(Correction 2026-10-06: an earlier wording of this paragraph called the mock
review "external"; it was not.)

## A. Additional retrieval arms (secondary)
1. **Reranking.** Qwen3-Reranker-0.6B (sentence-transformers CrossEncoder,
   `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`) reranks the hybrid top-30 for every
   chunk set; same metrics and budgets. Fallback if it cannot run: BAAI
   bge-reranker-v2-m3. Reported as "hybrid + rerank".
2. **Modern dense embedder.** qwen3-embedding:0.6b (Ollama) as a dense
   retriever and in a second hybrid (BM25 + qwen3-embedding, RRF k=60).
   (Already listed as secondary in plan v2; this makes the hybrid explicit.)
3. **Late chunking.** jina-embeddings-v3 token embeddings of each document
   (8,192-token windows), mean-pooled over the Stage-1 structural chunk spans,
   compared with plain jina-embeddings-v3 embeddings of S, S+T, E and CR.

## B. FreshStack answer-level outcome (secondary; replaces plan v2 sec. 5/8 "no judge")
- Generator: local qwen2.5:7b, 1,024 tokens of retrieved source text (hybrid),
  systems F512c, F256t, S, S+T, E, CR.
- Judge: meta-llama/llama-3.3-70b-instruct via OpenRouter pinned to DeepInfra,
  temperature 0; a different family from all generators and enrichers. Labels
  support / partial_support / no_support per nugget (TREC 2024 AutoNuggetizer
  scheme, as used by FreshStack); strict nugget recall is the outcome.
- Acceptance (as in plan v2 sec. 8): positive control (own Stack Overflow
  answer) >= 85% strict recall, negative control (another question's answer)
  <= 5%. If an author spot-check is completed, Cohen's kappa >= 0.6 is also
  required; without it, results are labelled exploratory.
- Margin for E vs S+T and E vs CR: +/-5 strict-recall points.

## C. Within-family scaling (exploratory)
E with Qwen3-235B-A22B-Instruct-2507 (DashScope, temperature 0, thinking off)
on a random 30 of the 300 Qasper v2 papers (`random.Random(2027).sample`),
compared with E (local qwen2.5:7b) and E-strong (gpt-oss-120b) on the same
papers. Separate frozen configuration file; no effect on primary arms.

## D. Pilot erratum (correction, not a new analysis)
The v3 pilot used laravel/docs commit 1e8496c2; FreshStack's corpus matches
724c31cc. The pilot is re-scored on the correct commit; only the changed files
are re-built. Reported as a correction to v3.
