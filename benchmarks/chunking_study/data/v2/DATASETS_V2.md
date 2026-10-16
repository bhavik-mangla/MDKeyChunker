# v2 confirmatory datasets

Built 2026-10-06 by `prepare_v2_qasper.py` and `prepare_v2_freshstack.py` (repo root).
Neither script calls an LLM or an API. Raw downloads are cached in `data/v2/raw/`
(168 MB: FreshStack corpus/query parquets, GitHub trees, raw .md files).
Structural chunks are `len(MarkdownChunker(Config()).chunk(md))`. Tokens use
tiktoken `cl100k_base`. "CR input tokens" is the sum over chunks of the
whole-document token count, i.e. the prompt volume of contextual retrieval
before any KV-cache reuse.

## 1. Qasper v2 (`qasper_v2.json`)

Rules:
- Pool: `data/qasper-dev.parquet` (281) + `data/qasper-test.parquet` (416) = 697 papers,
  minus the 30 pilot ids in `data/subset.json` = 667.
- Eligible: at least one question with text evidence. Evidence that starts with
  `FLOAT SELECTED` is dropped. Evidence containing `:::` is also dropped: these
  are section-name paths such as "Method ::: Encoder", not paragraphs. This filter
  is new in v2; prepare.py did not apply it. 236 such strings exist across dev+test.
  Evidence from annotators who marked the question unanswerable is ignored, as in v3.
  13 papers have no such question, which leaves 654 eligible.
- Sample: the pool is sorted by paper id, then `random.Random(2027).sample(pool, 300)`.
- Markdown conversion: `prepare.to_markdown`, unchanged from v3.
- Per question: `evidence` is the union over annotators. `evidence_by_annotator`
  holds one filtered list per annotator, aligned with `answers`. `answers` has one
  entry per annotator, including unanswerable ones, as the official evaluator
  expects. Each entry has `type` (extractive/abstractive/boolean/none, typed as
  `qasper_evaluator` does), `text`, raw `extractive_spans`, `free_form_answer`,
  `yes_no`, `unanswerable`, `annotation_id` and `worker_id`. Each paper also carries `split`.

| | |
|---|---|
| Papers | 300 (dev 105, test 195) |
| Questions | 951 |
| Annotators per question | 1: 80, 2: 601, 3: 233, 4: 33, 5: 4 |
| Answer types, all annotator answers (2,133) | extractive 1,225, abstractive 510, boolean 283, none 115 |
| Answer types, first annotator | extractive 553, abstractive 237, boolean 123, none 38 |
| Evidence paragraphs per question | 1: 432, 2: 269, 3: 100, 4+: 150 |
| Markdown | 6,841 KB; 1.45 M tokens; 2 papers > 16k tokens (max 29.5k) |
| Structural chunks | **7,222** (24.1 per paper) |
| CR input tokens | 43.1 M |
| Evidence paragraphs found verbatim in Markdown | 2,231 / 2,245 (the remaining 14 differ only in whitespace or encoding; rec@ metrics should normalise) |

## 2. FreshStack v2 (`freshstack_<domain>.json`, `freshstack_v2_docs.json`)

### Commit identification
`freshstack/corpus-oct-2024` metadata stores only branch URLs (`.../blob/main/...`).
The HF card, paper and GitHub repo give no commit. Each repo was partially cloned
(`--filter=blob:none`). Every commit that touched the docs root between Oct and
Nov 2024 (Angular: Sep 2024 to Jan 2025) was then scored by how many of the
FreshStack corpus chunks under the docs root match the file text at that commit
exactly. All chunks were scored, not only gold ones.

| Domain | Repo, docs root | Commit (date) | Corpus chunks matched | Runner-up |
|---|---|---|---|---|
| laravel | laravel/docs (11.x), all `*.md` | `724c31ccd3edce6b6dfe5e0dd2a594a47217f078` (2024-10-24) | 455/455 | 452 |
| angular | angular/angular, `adev/src/content/**/*.md` | `7d9b38e97bbef0c6f6e3761e7cd86457c1e0f2e1` (2024-10-28) | 383/384 (2 relocated, 1 unlocatable) | 381; parent 63f3d0c8 ties, with identical Markdown (only CLI help JSON differs) |
| yolo | ultralytics/ultralytics, `docs/en/**/*.md` | `8d203cf40dadb9343241d2f65bd4c84820f27430` (2024-10-26) | 565/565 | 563 |

At each commit, the set of `.md` files under the root equals the set in the corpus
(98 / 283 / 335). FreshStack offsets are **character** offsets into the decoded
file. They equal byte offsets only when the preceding text is ASCII.

The one Angular mismatch is in `guide/image-optimization.md`. FreshStack stripped
a `data:image/png;base64,...` placeholder, so its later chunks are shifted by 33
characters. No version of the file in history matches. Two of those chunks are
located uniquely at shifted offsets ("relocated"). Chunk `_7202_10892` cannot be
located. No kept gold span is affected: every kept span is located at its stated
offsets (`located: offset_char`).

**The v3 pilot used the wrong Laravel commit.** v3 used `1e8496c2` (2024-10-30).
At that commit 26 corpus chunks do not match, in 7 changed files including
`eloquent-relationships.md`. In `data/freshstack_laravel.json`, 6 of 61 unique
gold spans differ from FreshStack's chunk text: 5 in eloquent-relationships.md
and 1 in releases.md. These affect 15 of 73 pilot questions. The shifts are
small, but the gold spans are not byte-exact against FreshStack. The pilot data
was not changed.

### Rules
- Corpus: **every** `.md` file under the docs root at the commit, not a
  gold-picked subset. Files come from raw.githubusercontent.com at the commit,
  and each file is checked against the git blob SHA-1 in the commit tree.
- Gold is the union, over nuggets, of `relevant_corpus_ids` (queries-oct-2024, test).
  A span is kept when its corpus chunk text equals `markdown[start:end]`.
  Non-Markdown spans and Markdown spans outside the docs root (for example
  `angular/CHANGELOG.md`, `tools/manual_api_docs`, framework READMEs) are dropped
  from the gold set and counted per question in `gold_dropped`.
  `all_gold_markdown_in_root` flags questions that lost nothing. Questions with no
  kept span are dropped (`dropped_no_markdown_gold`).
- **Laravel pilot questions are excluded** (73 qids, `excluded_pilot_qids`), per
  ANALYSIS_PLAN_v2 §1. Without this exclusion Laravel would keep 162 questions.
- Question fields: `qid`, `title`, `question` (title + HTML-stripped body, as in v3),
  `nuggets` (text), `nugget_evidence` (indices into `evidence` per nugget),
  `evidence` [{path, start, end (char), start_byte, end_byte, corpus_id, located, text}],
  `gold_dropped`, `answer_id`, `answer` (accepted Stack Overflow answer, present
  for all kept questions) and `tags`.
- Verification: the full check found every kept gold span byte-exact (`raw[start_byte:end_byte] == corpus text`)
  in all 3 domains. A seeded random 12 per domain were re-checked separately: 12/12 in each.

| | laravel | angular | yolo | total |
|---|---|---|---|---|
| Files (all .md under the root) | 98 | 283 | 335 | 716 |
| KB | 2,808 | 1,810 | 2,647 | 7,264 |
| Doc tokens | 644k | 387k | 622k | 1.65 M |
| Docs > 16k tokens | **8** | 0 | 0 | 8 |
| Structural chunks | 3,860 | 2,307 | 3,893 | **10,060** |
| CR input tokens | **48.2 M** | 7.0 M | 11.3 M | 66.5 M |
| Questions total / no Markdown gold / pilot-excluded / **kept** | 184 / 22 / 73 / **89** | 129 / 30 / 0 / **99** | 57 / 10 / 0 / **47** | **235** |
| Kept with all gold in root Markdown (no gold dropped at all) | 14 | 7 | 4 | 25 |
| Kept with all *Markdown* gold in root (non-Markdown gold may be dropped) | 74 | 51 | 38 | 163 |
| Kept gold spans (unique) | 302 (180) | 295 (146) | 209 (105) | |
| Gold files | 71 | 118 | 68 | 257 |
| Unique spans with code fence / table / list | 130 / 10 / 126 | 51 / 46 / 81 | 87 / 26 / 71 | |
| Gold dropped in kept questions: non-Markdown / Markdown outside root | 398 / 47 | 825 / 125 | 311 / 22 | |

`freshstack_v2_docs.json` holds 716 docs `{id: "<domain>__<path with / -> __>", domain, path, markdown}`.

## 3. Flags and cap proposal (no cap applied)

1. **Chunk budget exceeded.** Qasper has 7,222 chunks and FreshStack has 10,060,
   for **17,282** in total against a guide of about 12k. A file-level cap cannot
   reach 12k without dropping questions. The gold files alone hold 5,675 chunks
   (Laravel 3,231: 71 of its 98 files are gold. Angular 1,302. YOLO 1,142.), so
   Qasper plus gold-only FreshStack is already 12.9k.
   Proposed cap (B). Keep all gold files plus a seeded random sample
   (`random.Random(2027)`, sorted paths) of non-gold files equal to 0.5 times the
   number of gold files. Apply it per domain to Angular and YOLO only. Laravel's
   whole corpus already has a 0.38 distractor ratio, so it stays full.
   Result: Angular 177 files / 1,694 chunks, YOLO 102 / 1,498, Laravel 98 / 3,860.
   FreshStack total 7,052, overall **14.3k**. A ratio of 1.0 gives 2,018 + 1,915,
   overall 15.1k.
   Caveat: a gold-aware corpus reintroduces the selection the full corpus was
   meant to remove (threat T4). It is the same for every arm, so paired
   differences remain valid, but it makes retrieval easier. Recommendation: keep
   the **full corpora** for E (local), and apply cap B only to arms whose cost
   scales with document length (CR).
2. **Laravel CR cost.** 8 Laravel docs exceed 16k tokens (`NUM_CTX` default is
   16384; v3 FreshStack used 32768). No doc exceeds 32k. Laravel CR input is
   48M tokens, 72% of the FreshStack CR total. Ollama's prefix KV-cache reuse per
   document cuts the real prefill a lot, but plan CR-strong on Laravel carefully.
3. **Power.** 235 FreshStack questions remain after excluding the pilot, against
   the ~330 the power analysis assumed. That analysis counted Laravel 168 with
   pilot questions included. ±3 TOST power on hybrid will be below the planned
   0.83. Re-run `scratchpad/power.py` with n=235, or reconsider the pilot exclusion.
4. **Gold-in-root rule.** The design note's rule ("all Markdown gold lies
   within the included directories") would keep 163 questions (74 / 51 / 38).
   Requiring that no gold at all is dropped would keep 25. This build uses the
   ≥1-span rule from the brief. Both subsets can be filtered from `gold_dropped`.
