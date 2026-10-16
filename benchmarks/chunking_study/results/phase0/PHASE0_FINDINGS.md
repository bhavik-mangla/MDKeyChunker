# Phase 0 findings: zero-LLM-cost re-analyses of the v3 PILOT data

Computed 2026-10-06. Everything here uses the existing v3 caches: Qasper has 30 papers and 79 questions, FreshStack Laravel has 24 docs and 73 questions. No LLM calls were made. **All results are PILOT and exploratory.** They were not pre-registered, and only A6 restates pre-registered cells.

Units: differences are in points (x100). The primary metrics are Qasper rec@512t and FreshStack prec@1024t. CIs are paired percentile cluster-bootstrap intervals (clusters are papers for Qasper and questions for FreshStack). A1–A5 use 10k resamples and the v2-style `analyse()`.

Sanity checks: the harness in `phase0/common.py` reproduces `results/qasper_per_question.json` exactly (struct, enr_rk and fixedtok on all 4 retrievers). The re-implemented BM25 gives the same rankings as `rank_bm25`.

Code is in `phase0/`. Per-analysis tables are in `results/phase0/a*_*.md` and the full numbers in `a*_*.json`. New embeddings are cached in `data/emb_phase0/`; `data/emb` and `data/cache` were only read.

## A1. Qasper structure confound (`a1_structure.md`, `a1c_window_phase.json`)

**(a) Greedy-oracle rec@512t ceiling.** Every chunking allows about 0.94–0.99: F512c 0.993, F256t 0.949, S 0.968, paragraph-packer 0.941, merged_rk 0.951. Actual scores are 0.25–0.50, so no chunking is limited by its ceiling. RQ1 gaps come from how well each chunking ranks, not from what it can reach.

**(b) Evidence paragraphs split across windows** (143 unique evidence paragraphs of 5 words or more):

| Chunking | Not whole in one chunk | Less than 50% in any one chunk |
|---|---|---|
| F512c | 83.9% | 14.0% |
| F256t | 37.1% | 0% (32-token overlap) |
| Length-matched 203-token window | 53.1% | 0% |
| S and the paragraph packer | 0% | 0% |

**(c) Structure-blind controls.**
- **Paragraph packer.** Whole paragraphs up to 1,500 characters, headers ignored. It scores the same as F256t: hybrid +0.3 [−11.1, +11.4], BM25 −1.6.
- **S still beats the paragraph packer.** Hybrid +12.4 [+5.1, +20.3], nomic +10.9 [+3.8, +18.0]. The gap stays at +9 to +11 (hybrid CI excludes 0) when the packer is shrunk to S's size (600–1,000 characters, median 115–167 tokens).
- **S also beats the length-matched window.** The window is 203 tokens, the median S chunk, averaged over 4 phases. Gaps: BM25 +8.0 [+1.9, +14.2], nomic +13.8, hybrid +15.4 [+8.8, +22.8].
- **Caveat: fixed windows are fragile.** The BM25 score of one fixed-window configuration moves between 0.20 and 0.32 with window size and phase alone. A single window setting is a noisy baseline. The single-phase 203-token window scored 0.199 on BM25.

**Conclusion:** keeping paragraphs whole does not explain RQ1. The gain follows **header-bounded sections**. The reviewer's "alignment artefact" objection is not supported on this pilot.

**(d) Max-over-annotators vs union evidence.** 32 of 79 questions have at least 2 distinct annotator evidence sets. No conclusion changes:
- RQ1 differences grow (hybrid S−F256t goes from +12.7 to +15.4).
- RQ2 and RQ3 stay non-significant.
- RQ4b BM25 rk−nork goes from −6.0 to −7.6, with CI still excluding 0.
- merged_rk−enr_rk under hybrid moves from +2.4 [−0.2, +5.9] to +3.2 [+0.6, +6.6], so it now excludes 0.

## A2. BM25 IDF scope and k1/b (`a2_bm25.md`)

- **BM25 alone.** E−S+T is positive in every one of the 12 settings: per-paper or corpus IDF, k1 ∈ {0.9, 1.2, 1.5}, b ∈ {0.4, 0.75}. The range is +4.1 to +8.8, and the CI excludes 0 in 2 of 12 settings. E−CR ranges from −2.8 to +4.5 and is never significant. So the exploratory "LLM prefixes help BM25" signal survives corpus IDF. It is mostly a lexical-expansion effect (see A3).
- **Hybrid is sensitive to the BM25 setup.** With corpus IDF, hybrid E−S+T flips from −2.3 to +5.3 [−1.4, +13.4] and E−CR from −4.1 to +3.9. Both remain non-significant.

## A3. Field ablation (`a3_fields.md`)

- **Qasper BM25.** The generated **questions** carry most of the gain: S+q−S = +6.1 [−0.6, +13.1], against E−S = +8.0. Title+summary gives +2.5 and keywords +0.0.
- **Qasper hybrid.** Single fields hurt: title+summary −7.0 [−12.6, −1.5], keywords −4.7 [−9.0, −0.7].
- **FreshStack.** No single field matters on BM25 or hybrid; all are within ±2.
- **FreshStack mxbai.** E−S = +4.9 [+0.7, +9.1]. Questions (+3.6) and keywords (+3.2) each give part of it; neither alone is significant.

## A4. RQ4b random-merge control (`a4_random_merge.md`)

The control makes the same number of merges as the key-based merger (94 on Qasper, 56 on FreshStack), chosen at random within each document. There are two versions, merging any two chunks or only adjacent ones, each with 5 seeds.

- **Qasper hybrid.** Key-based merging beats random merging: +4.8 [+0.5, +9.7] against any-pair merges and +4.4 [+0.9, +7.9] against adjacent merges. Nomic shows +6.4 against any-pair merges. On BM25 it scores the same as random merging (about 0).
- **Seed spread is large.** Hybrid seed means range from 0.39 to 0.45.
- **FreshStack.** Every difference is within ±1.5.
- **Conclusion.** Keys choose better merges than chance on Qasper. Merging still does not beat no merging (merged_rk−enr_rk is ns in v3).

## A5. mxbai truncation hypothesis (`a5_truncation.md`)

Token counts use mxbai's own WordPiece tokenizer (HF `tokenizer.json`, run with the `tokenizers` package). Only 1–2.4% of S, S+T, CR and E chunks exceed 512 tokens on either dataset. About 2–2.6% of FreshStack source text is cut. Merged chunks are cut more: 6.6% of chunks on FreshStack and 11.4% on Qasper.

**Truncation cannot explain the FreshStack mxbai gain for E.** Moving the generated fields after the text (meta-last) lowers mxbai: −3.5 [−6.3, −0.6]. So the gain comes from metadata placed first steering mxbai's embedding, not from metadata surviving truncation.

**Qwen3-Embedding-0.6B (32k context) on FreshStack:**
- E−S+T = −3.3 [−8.1, +1.3]. With BM25+qwen3 hybrid it is −0.5.
- E(meta-last) beats E by +6.2 [+2.2, +10.3].

**Qwen3-Embedding-0.6B on Qasper:**
- E−S+T = +9.2 [+0.1, +17.5].
- CR scores best (0.553), and E−CR = −1.4.

The direction of the E effect therefore flips with the embedder and with where the metadata is placed. It is embedder-specific, not a general enrichment benefit.

## A6. v2-style statistics for the pre-registered v3 cells (`a6_stats.md`)

Method: 10k sign-flip permutations, 95% and 90% CIs, Holm adjustment, and TOST at ±5 (Qasper) and ±3 (FreshStack).

- **No RQ2 or RQ3 cell is equivalent** on any retriever and either dataset. Example: Qasper hybrid E−S+T has a 90% CI of [−7.6, +3.2]. All RQ2/RQ3 cells are **inconclusive**. Holm-adjusted TOST p in the v2 primary family is 0.865 for every test.
- **What survives Holm.** Only Qasper hybrid S−F256t (Holm p = 0.036 across all 22 primary cells) and S−F512c (0.002) survive. FreshStack hybrid S−F512c survives only in the 4-test v2 RQ1 family (0.020); it does not survive across all 22 cells (0.196).
- **The v3 "significant" RQ4b BM25 cell** (rk−nork −6.0) has permutation p = 0.092. The percentile bootstrap CI excluded 0 but the permutation test does not reject, and Holm p = 1.0.
- **FreshStack merged_rk−enr_rk is equivalent** within ±3 on all 4 retrievers.

## What changes for the paper

1. **Rephrase the RQ2/RQ3 nulls.** Say "inconclusive; the pilot cannot exclude ±5 (Qasper) or ±3 (FreshStack)", not "no difference". This motivates the v4 confirmatory samples.
2. **RQ1 can now answer the alignment critique.** The structure gain survives paragraph-preserving and length-matched controls, so attribute it to header-bounded sections. Also report that fixed-window baselines vary by about ±6 points with window phase.
3. **Demote the RQ4b BM25 "worse" result.** It does not survive a permutation test or Holm. A positive point to add: key-based merges beat random merges on Qasper hybrid.
4. **Retract or rephrase the mxbai truncation explanation.** Truncation is too rare, and metadata-first ordering drives the effect. The E gain is embedder- and position-specific (it reverses with Qwen3-Embedding).
5. **Explain the BM25 gain as doc2query-like.** The generated questions drive it, and it is robust to IDF scope and k1/b. Hybrid conclusions are sensitive to the BM25 configuration.

Not done: a reranker. The analyses did not require one, and none was tested.
