# Analysis plan v2: confirmatory study (DRAFT — becomes binding when pushed)

Status: draft written 2026-10-06 before any v2 build. It becomes binding at the
commit that removes the word DRAFT from this title; that commit is pushed to
GitHub before the first v2 build, so GitHub's server-side push time is the
external timestamp. The v3 study (ANALYSIS_PLAN.md) is treated as a pilot. Its
data and the Phase 0 re-analyses of it (results/phase0/) informed this plan and
are excluded from every confirmatory analysis below.

## 1. Samples
- **Qasper v2:** 300 papers sampled with `random.Random(2027)` from dev + test
  papers that are not among the 30 pilot papers and have at least one question
  with text evidence (same evidence rules as v3). The paper list is committed
  in data lists before any build (`data/v2/qasper_v2.json` ids).
- **FreshStack v2:** Laravel, Angular and YOLO test questions whose gold
  includes at least one Markdown span located byte-exactly in the full docs
  corpus of that domain at the commit FreshStack used. The 73 Laravel questions
  used in the pilot are excluded. Corpora, commits and question lists are
  committed before any build.

## 2. Arms (chunk sets x index text)
F512c (512-char windows), F256t (256-token windows, 32 overlap), S (structural),
S+T (structural + title chain), E (enrichment, local qwen2.5:7b), CR
(contextual retrieval, local qwen2.5:7b), E-strong and CR-strong
(openai/gpt-oss-120b via OpenRouter, provider-pinned, reasoning effort low,
temperature 0, seed 0). Prompts, model ids, provider pins and quantization are
frozen by hash in `results/v2/frozen_config.json` before any build.

## 3. Primary outcomes and retriever
- Qasper: evidence recall within 512 tokens of source text (rec@512t),
  within-paper retrieval, cluster = paper.
- FreshStack: gold-token precision within 1,024 tokens (prec@1024t), pooled
  retrieval per domain, cluster = question.
- Primary retriever: hybrid (BM25 + nomic-embed-text, RRF k=60).
  **Change from v3:** BM25 is now secondary (v3 had it co-primary), to reduce
  multiplicity. This change was made after viewing pilot results and is
  disclosed as such.

## 4. Hypotheses
Equivalence margins: +/-5 points on Qasper rec@512t; +/-3 points on FreshStack
prec@1024t (chosen from pilot paired SDs for >= 0.8 power at the planned sizes).
- **H1** E vs S+T and **H2** E vs CR, local model, hybrid retriever, each dataset:
  TOST equivalence with the margins above (90% paired cluster-bootstrap CI,
  10,000 resamples, BCa). Primary family = these 4 tests, Holm-adjusted.
  Verdicts: equivalent (90% CI within margin), different (95% CI excludes 0),
  inconclusive (otherwise).
- **H3** (RQ1) S vs F512c and S vs F256t, hybrid, each dataset: two-sided
  difference tests; separate family of 4, Holm-adjusted.

## 5. Key secondary (Holm within this family)
- H1 and H2 with E-strong / CR-strong.
- Qasper answer token-F1 (official evaluator, max over gold answers) from a
  fixed local generator given exactly 512 tokens of retrieved source text.
- FreshStack strict nugget recall (FreshStack protocol) from a fixed generator
  given 1,024 tokens of retrieved source text, judged by a model of a different
  family from the generator and enrichers.

## 6. Secondary / exploratory (unadjusted CIs, labelled)
BM25, mxbai-embed-large, qwen3-embedding; budgets 256/1,024/2,048; pooled-corpus
Qasper; field ablations; RQ4 descriptive statistics; robustness runs (repeat
generations, paraphrased prompt); judge-based Qasper correctness; per-domain
FreshStack results.

## 7. Exclusions (fixed in advance)
- A question is dropped before any run if its gold cannot be located
  byte-exactly in the corpus text.
- LLM failures: up to 3 retries, then the chunk keeps empty fields and is
  counted, never dropped.
- A document that fails to build in any arm is dropped from all arms (paired
  completeness) and counted; if more than 5% of documents fail, this is
  reported and nothing is replaced.
- No outlier removal. No change of budgets, retrievers, margins or metrics
  after outcome metrics are viewed.

## 8. LLM judge acceptance
The judge is accepted if (a) positive controls (gold answers) score >= 85%,
(b) negative controls (answers to a different question) score <= 5%, and
(c) the author's blind spot-check of about 120 items reaches Cohen's kappa
>= 0.6 with the judge. A second judge (different family) scores 25% of items
for an agreement estimate. If the judge fails, judge-based outcomes are
reported as exploratory only; Qasper answer-F1 is unaffected.

## 9. Stopping and monitoring
No interim looks at outcome metrics. During builds only cost, parse-failure
rate and throughput are monitored.

## 10. Budget
Paid API spend capped at $5.50 by a ledger-enforced guard. If the cap would be
exceeded, the strong-model arm is reduced in this pre-declared order:
(1) drop CR-strong on FreshStack beyond Laravel, (2) restrict CR-strong on
Qasper to a random 150 of the 300 papers (seed 2027).

## 11. Deviations
Any deviation is logged with date and reason in `DEVIATIONS.md` and reported in
the paper.
