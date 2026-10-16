# Erratum: wrong laravel/docs commit in the v3 PILOT FreshStack evaluation

Scope: the v3 pilot only (`data/freshstack_laravel.json`, `results/freshstack*.json`,
`results/FRESHSTACK_FINDINGS.md`). v2 data already uses the right commit (`data/v2/DATASETS_V2.md` §2).

## What was wrong

There were two defects in `prepare_freshstack.py`:

1. **Commit.** The pilot pinned laravel/docs `1e8496c2` (2024-10-30). The FreshStack corpus matches `724c31cc` (2024-10-24).
2. **Offset unit.** FreshStack corpus-id offsets are *character* offsets. The pilot sliced them as *bytes*. The two agree only when no non-ASCII text comes before the span.

Effect on the pilot's 24 files and 61 unique gold spans:

| | file text | gold spans not exact vs FreshStack chunk text | cause |
|---|---|---|---|
| `eloquent-relationships.md` | differs: 1 line, 94,087 → 94,085 chars. Line 1075 reads `morphMany` → `hasMany` | 5 spans. 4 are shifted 2 chars; `_43015_49424` differs by 4 chars | commit |
| `releases.md` | identical at both commits | 1 span (`_2381_11842`), 2 chars short | byte vs char offsets: 2 non-ASCII chars come before it |
| other 22 files | identical | 0 | |

The 6 bad spans occur in 17 evidence entries across **15 of 73 questions**:
75915323, 76436680, 76721334, 77038806, 77580974, 77969870, 77977254, 78020564,
78188214, 78357045, 78366994, 78429920, 78435847, 78479954, 78490834.

## Correction

`phase0/erratum_data.py` writes `data/freshstack_laravel_fixed.json`. It keeps the same 24 files and 73 questions. File text is taken from `data/v2/raw/laravel/files/`, and each file is verified against the git blob SHA-1 in the `724c31cc` tree. Spans are re-sliced by character offset.

- Gold spans that match the FreshStack corpus chunk text exactly: **61/61** (pilot: 55/61).
- Each changed span is marked `evidence[].changed`.

## Zero-LLM systems (fixed512, fixedtok, struct/text, struct/tc)

`phase0/erratum_chunks.py` rebuilds chunks into `data/cache_fixed/`. The other 23 files are copied from `data/cache/`. As a check, all 24 × 3 chunk sets were first rebuilt from the old text, and they came out identical to the pilot cache.

For `eloquent-relationships.md`, the following chunks changed:

| chunk set | chunks changed |
|---|---|
| fixed512 | 93/184 (boundaries shift) |
| fixedtok | 48/93 |
| struct | 1/96 (chunk 45) |

`phase0/erratum_eval.py` re-runs the unmodified `evaluate_corpus.main` with the cache root and data path swapped. Embeddings are cache-aware: 286 new vectors were written to `data/emb_phase0`. The wrapper reproduces `results/freshstack.json` exactly (572/572 entries) from the pilot cache and data.

Corrected results are in `results/phase0/erratum_freshstack.json` (+ `_per_question`, `.txt`). The comparison with the published numbers is in `results/phase0/erratum_compare.{json,txt}`.

**Per-system means**, 4 systems × 4 retrievers × 11 metrics = 176 cells:

| system | max \|Δ\| | cells with \|Δ\| ≥ 0.005 |
|---|---|---|
| fixed512/text | 0.027 | 14 |
| fixedtok/text | 0.014 | 9 |
| struct/text | 0.0001 | 0 |
| struct/tc | 0.0000 | 0 |

The largest change is nomic fixed512 hit@10, which equals 2 questions. On prec@1024t no cell moves by more than 0.010 (nomic fixedtok, 0.346 → 0.355).

Per-question values moved for 43 questions: 13 of the 15 affected ones and 30 others. The 30 others moved because the fixed windows shifted, which changes rankings.

**Pre-registered RQ1 comparisons**, 88 cells: max |Δ| 0.027, and no change of sign in any estimate that the findings rely on. Headline prec@1024t (published → corrected):

| comparison | retriever | published | corrected |
|---|---|---|---|
| struct − fixedtok | hybrid | −0.026 [−0.064, +0.012] | −0.023 [−0.062, +0.014] |
| struct − fixedtok | nomic | −0.008 | −0.018 |
| struct − fixedtok | mxbai | −0.010 | −0.014 |
| struct − fixedtok | bm25 | −0.022 | −0.022 |
| struct − fixed512 | hybrid | +0.051 [+0.014, +0.090] | +0.049 [+0.012, +0.088] |
| struct − fixed512 | nomic | +0.065 | +0.069 |
| struct − fixed512 | mxbai | +0.034 | +0.034 |
| struct − fixed512 | bm25 | +0.030 | +0.030 |

Two significance flips, both at a CI bound within ±0.0005 of 0:

- BM25 struct − fixed512 prec@1024t: lower bound −0.0003 → **+0.0002**. FINDINGS line 7 says "BM25/mxbai CIs touch 0"; that is now true only for mxbai.
- Hybrid struct − fixed512 rec@512t: lower bound +0.0005 → −0.0004 (now not significant).

## LLM systems (enr_rk, merged_rk, cr): rebuild pending

Of the 96 struct chunks in `eloquent-relationships.md`, only chunk 45 changed ("…when defining a `hasMany` relationship"). That still requires rebuilding the whole doc:

- **enr_rk:** rolling keys pass from chunk to chunk, so chunks 45–95 can change.
- **cr:** every CR prompt contains the full document, so all 96 chunks can change.
- **merged_rk:** follows from the rebuilt enr_rk.

`data/cache_fixed/REBUILD_LIST.json` lists 1 doc and 192 LLM calls: 96 for enr_rk and 96 for cr, with qwen2.5:7b, NUM_CTX=32768, seed 0, temperature 0. In the pilot this doc took about 58 min (enr_rk) + 23 min (cr).

`build.py` hard-codes `CACHE = data/cache`. The rebuild must redirect it to `data/cache_fixed`, and copy the other 23 docs' enr_rk/merged_rk/cr caches from `data/cache`.

**Questions that could be affected: up to 44 of 73.**

| reason | questions |
|---|---|
| gold span changed | 15 |
| gold in eloquent-relationships.md | 16 |
| an eloquent-relationships chunk falls in the top-10 or the 2048-token scoring window of any LLM system and retriever | 41 |
| union | 44 |

**Provisional estimate** (`results/phase0/erratum_freshstack_provisional.json`): this uses the stale pilot LLM chunks for that one doc with the corrected gold. Every RQ2–RQ4 comparison (132 cells) moves by less than 0.0001 and none flips significance. The final LLM-system numbers must wait for the rebuild.
