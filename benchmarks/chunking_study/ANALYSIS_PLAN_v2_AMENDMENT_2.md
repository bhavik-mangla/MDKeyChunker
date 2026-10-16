# Amendment 2 to analysis plan v2: rolling keys and key-based merging (RQ4)

Date: 2026-10-08. Written after the plan v2 primary and key-secondary
retrieval results (H1-H3) were computed, and **before any rolling-key or
merged arm was built or scored in v2**. No v2 outcome involving Stage 3,
merged chunks, or enrichment without the rolling key dictionary exists at the
time of writing. Plan v2 sections 1-5 and Amendment 1 are unchanged.

Plan v2 section 6 already lists "RQ4 descriptive statistics and key-based vs
random merging" as secondary. This amendment makes the rolling-key question a
pre-specified family of its own, adds two controls, and fixes the tests.
Because it is written after H1-H3 were seen, the family is labelled
**pre-specified secondary**, not primary.

## A. Arms (all on the plan v2 samples: 300 Qasper papers, 377 FreshStack docs)
- **M**: MDKeyChunker Stage 3 (`Restructurer`, package defaults) applied to
  the cached E chunks (local qwen2.5:7b). No new LLM calls. Index text = meta.
- **E-noRK**: enrichment with the same prompt and model as E, but the rolling
  key list is replaced by "(not provided)" on every call (the pilot's
  `NoRollingKeysEnricher`; the first-chunk placeholder would contradict the
  chunk position and previous summary in the same prompt).
  (Correction 2026-10-08, before any build: the first pushed wording named the
  first-chunk placeholder; the pilot ablation and this study use "(not provided)".) Same frozen model, options and
  exclusion rules as E (plan v2 sec. 7).
- **M-noRK**: Stage 3 applied to E-noRK.
- **RM**: random-merge control. Per document, the same number of merges as M
  (len(E) - len(M)), merging random adjacent pairs under the same
  `max_merged_size`; 5 seeds, per-question scores averaged over seeds.
  Merged chunks carry the concatenated metadata, as in M.
- **SM**: semantic-merge control. Per document, the same number of merges as
  M, chosen greedily as the adjacent pair with the highest cosine similarity
  of nomic-embed-text chunk embeddings (text only), under the same
  `max_merged_size`. Metadata concatenated as in M.

## B. Hypotheses (family H4; hybrid retriever; plan v2 primary metrics and
margins: Qasper rec@512t +/-5, FreshStack prec@1024t +/-4; paired cluster
bootstrap BCa and sign-flip permutation as in plan v2; Holm over the 8 tests)
- **H4a (keys vs chance)**: M - RM, two-sided sign-flip test.
- **H4b (keys vs embedding similarity)**: M - SM, two-sided sign-flip test.
- **H4c (does Stage 3 cost or help retrieval)**: M - E, TOST at the plan
  margins (Holm over p_tost), with the two-sided result also reported.
- **H4d (does the rolling dictionary help enrichment)**: E - E-noRK, TOST at
  the plan margins (Holm over p_tost), two-sided result also reported.

Each hypothesis is tested on both datasets (2 x 4 = 8 tests). Decision rules
as in plan v2 sec. 4: "different" if the 95% CI excludes 0 and Holm-adjusted
p < 0.05; "equivalent" if Holm-adjusted p_tost < 0.05; otherwise
"inconclusive".

## C. Descriptive mechanism statistics (no tests)
Per dataset, E vs E-noRK: key reuse rate (share of chunks whose key appeared
earlier in the document), distinct keys per document, chunks sharing a key,
merges performed by Stage 3, merged-chunk length distribution and share of
merged chunks over the retriever's 512-token input.

## D. Exploratory (unadjusted, labelled)
M-noRK - M; M - S+T; all H4 contrasts under BM25, nomic and mxbai; per-domain
FreshStack results.

## E. Exclusions
As plan v2 sec. 7. A document that fails in E-noRK is dropped from every H4
contrast and counted; the H1-H3 results are not recomputed.
