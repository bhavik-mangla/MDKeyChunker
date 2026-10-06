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
  includes at least one Markdown span located byte-exactly in the docs corpus
  of that domain at the commit FreshStack's corpus matches (laravel 724c31cc,
  angular 7d9b38e9, yolo 8d203cf4; verified by matching every corpus chunk).
  Non-Markdown gold spans are removed from the gold set and counted. The 73
  Laravel questions used in the pilot are excluded (235 questions remain).
  Corpus: all Markdown files for Laravel; for Angular and YOLO all gold files
  plus random distractor files (random.Random(2027)) equal to half the number
  of gold files, to fit local compute. The selection (data/v2/selection.json)
  is committed before any build. Note: the pilot used Laravel commit 1e8496c2,
  under which 6 of its 61 gold spans differ; this is an erratum to v3.

## 2. Arms (chunk sets x index text)
F512c (512-char windows), F256t (256-token windows, 32 overlap), S (structural),
S+T (structural + title chain), E (enrichment, local qwen2.5:7b), CR
(contextual retrieval, local qwen2.5:7b), E-strong (openai/gpt-oss-120b via
OpenRouter, provider-pinned, reasoning effort low, temperature 0, seed 0) on
both datasets, and CR-strong on Qasper only (CR-strong on the FreshStack
corpora exceeds the budget). Secondary control arms added after the pilot
re-analysis (Phase 0): P (paragraph packing that ignores headers, Stage-1 max
size), FL (token windows of the median structural-chunk length), and a
random-merge control for key-based merging (5 seeds). Prompts, model ids, provider pins and quantization are
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
Equivalence margins: +/-5 points on Qasper rec@512t (300 papers); +/-4 points
on FreshStack prec@1024t (235 questions). Chosen from pilot paired SDs for
power >= 0.85 at these sizes when the true difference is 0 (a +/-3 margin
would give about 0.65 power at n = 235).
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
- (No LLM judge is used in v2; FreshStack is evaluated at the retrieval level
  only.)

## 6. Secondary / exploratory (unadjusted CIs, labelled)
BM25, mxbai-embed-large, qwen3-embedding-0.6b; budgets 256/1,024/2,048;
pooled-corpus Qasper; field ablations; S vs P and S vs FL; RQ4 descriptive
statistics and key-based vs random merging; robustness runs (repeat
generations, paraphrased prompt); per-domain FreshStack results.

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

## 8. LLM judge
Not used in v2 (see section 5).

## 9. Stopping and monitoring
No interim looks at outcome metrics. During builds only cost, parse-failure
rate and throughput are monitored.

## 10. Budget
Paid API spend capped at $5.50 by a ledger-enforced guard. If the cap would be
exceeded, the strong-model arm is reduced in this pre-declared order:
(1) restrict CR-strong on Qasper to a random 150 of the 300 papers (seed 2027),
(2) restrict E-strong on FreshStack to the Laravel domain.

## 11. Deviations
Any deviation is logged with date and reason in `DEVIATIONS.md` and reported in
the paper.
