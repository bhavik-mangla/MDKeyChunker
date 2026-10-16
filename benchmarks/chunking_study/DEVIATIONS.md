# Deviations from analysis plan v2 (and Amendment 1)

## 2026-10-06 — Scaling arm sample truncated by free quota (Amendment 1, section C)
The exploratory within-family scaling arm (E with Qwen3-235B-A22B-Instruct-2507 on a
random 30 of the 300 Qasper v2 papers) is limited by the DashScope free quota (~950k
tokens per model). The run stops cleanly at the token cap; the projected completion is
~25-26 of the 30 papers. The stopping point depends only on processing order and token
usage, not on any outcome, so the comparison is restricted to the completed papers
(paired against the same papers for the other arms). No billing beyond the free quota
was used.

## 2026-10-06 — Interim outcome view during a smoke test (plan v2 sec. 9)
While testing the secondary-arm code (Amendment 1, section A), a smoke run printed
rec@512t for one Qasper v2 paper (1706.01875; 4 questions) for systems S and S+T
only, under hybrid, qwen3-embedding and hybrid-qwen3 retrieval. No enrichment (E),
contextual retrieval (CR) or other primary-hypothesis system was involved, and no
paired comparison was computed. The output was deleted. A `--no-stats` mode was
added so later smoke and production runs write per-question dumps without computing
or printing statistics until all builds are complete.

Update (2026-10-06 15:10): the scaling run stopped at the token cap with 23 of the
30 sampled papers complete (945,707 of 950,000 tokens; 0 failures). The scaling
comparison uses these 23 papers.

## 2026-10-06 16:50 — FreshStack local build split into an enrichment pass and a CR pass
Throughput only; no change to any arm, prompt, model or decoding setting. Running all
FreshStack local variants in one run requires num_ctx 32768 (for CR) and, with memory
limits, only 4 parallel slots (~39 h projected). Enrichment prompts need far less:
the worst case over all 7,052 FreshStack chunks is 4,657 prompt tokens + 1,000 output
(cl100k), so the enrichment pass runs at num_ctx 8192 with 8 slots and the CR pass at
num_ctx 32768 with 4 slots (~19-20 h projected). The context size only allocates
memory; prompts fit, so generated outputs are not affected. No FreshStack local output
existed before this change. frozen_config.json gains models.local.num_ctx.freshstack_enr
= 8192; the original is kept as frozen_config.orig.json.

## 2026-10-12 06:15 — qwen3-embedding secondary arm rerun into a separate folder
Operational only. The first run (2026-10-11 21:51) stopped with a connection error because
the default Ollama server (:11434) was down at that moment; no output was written. It was
rerun with identical settings while the reranker arm was running, writing to
results/v2/extra_qwen3/ to avoid two processes merging into the same dump, and merged into
results/v2/extra/ afterwards. Runs were paused by the user three times (2026-10-10 to
2026-10-12, SIGSTOP/SIGCONT); paused processes resume where they stopped, so no outputs change.

## 2026-10-12 06:20 — pilot erratum re-scored after the rebuild
The rebuilt eloquent-relationships.md (enr_rk, merged_rk, cr) was combined with the unchanged
pilot docs (copied from data/cache, as REBUILD_LIST.json specifies) and scored into
results/phase0/erratum_freshstack_final.json. The earlier erratum_freshstack.json (zero-LLM
systems only) and the provisional file are kept.

## 2026-10-12 07:00 — corrections from the independent verification (no new data)
An independent re-computation matched every reported difference (4 decimals) and CI
(within 0.003). It found reporting errors, corrected here; outcomes did not change:
1. Verdict labels now apply each family's multiplicity rule. Plan v2 families: "equivalent"
   needs Holm-adjusted p_tost < 0.05; "different" is the 95% CI excluding 0 (plan v2 sec. 4),
   with the Holm-adjusted sign-flip p also reported. Amendment 2-4 families: "different" also
   needs Holm-adjusted sign-flip p < 0.05 (Amendment 2 sec. B). Changes: Qasper M-RM
   "different" -> no difference detected (p_holm 0.078); Qasper E+KX - E+AX "different" -> no
   difference detected (p_holm 0.136); FreshStack E-strong - S+T "equivalent" -> inconclusive
   (Holm-adjusted p_tost 0.081).
2. The key-secondary family had 7 tests; plan v2 sec. 5 specifies answer F1 for E vs S+T and
   E vs CR only. The E-strong answer-F1 pairs moved to exploratory (5 tests; verdicts unchanged).
3. FreshStack answers: 4 judgements unparseable after 3 attempts had been dropped. Per plan v2
   sec. 7 they are now counted as no_support for every nugget and flagged (judge_failed). The
   judge's negative control failed (0.0675 > 0.05), so all FreshStack answer results are
   labelled exploratory, as Amendment 1 B requires.
4. Added the Amendment 3 statistic (share of chunks retrieved within the budget that have a key
   link) and the Amendment 4 calls/tokens comparison; per-field wall-clock seconds are no longer
   reported (E ran beside CR under different concurrency, so they are not comparable).

## 2026-10-12 07:00 — qwen3-embedding: context size
Ollama loads qwen3-embedding:0.6b with a 4,096-token context by default, and its runner crashed
("EOF") on 7 chunk texts of about 4,900 tokens in Qasper paper 1901.05415 instead of truncating
them as requested. The arm now loads the model with num_ctx 8,192 (checked: embeddings of shorter
texts are bit-identical) and cuts any document longer than 3,000 Qwen3 tokens client-side (the runner still crashed on some texts of 3,800-3,950 tokens at num_ctx 8,192). Batches
the runner rejects are retried one text at a time. Embeddings cached before the crash are unaffected.

Addendum (12:25): the run finished all 300 Qasper papers, then crashed on the first FreshStack
domain while embedding queries, which are whole Stack Overflow posts and were not cut. Queries now
get the same 3,000-token cut as documents. Qasper queries are far shorter, so its finished results
are unaffected. Only FreshStack is rerun (`--datasets freshstack`).

Addendum (13 Oct): the rerun still crashed on 3 FreshStack queries of 2,100-3,000 tokens. The
Ollama log shows the cause: the runner processes inputs in micro-batches of 2,048 tokens and crashes
on an embedding input that spans two. The arm now also sets num_batch 8,192 (one micro-batch per
input). Checked: embeddings of 120 cached chunk texts are bit-identical under the new setting, and
the 3 failing queries now embed. The 3,000-token cut is kept so Qasper and FreshStack use one rule.

## 2026-10-12 07:30 — planned robustness runs not run
Plan v2 sec. 6 listed robustness runs (repeat generations, a paraphrased prompt) as secondary.
They were not run: compute went to the mechanism tests of Amendments 2-4. Every LLM-built arm
is a single generation at temperature 0, seed 0; CIs cover question sampling only.

## 2026-10-12 07:30 — exploratory dial controls (post hoc)
After the H7 result, random adjacent merges (5 seeds) and most-similar adjacent merges were built
on the E-noRK base with the merge counts of M-TP (build_h4.py --tp-controls; evaluate_dial.py),
to ask whether M-TP's FreshStack gain comes from the amount of merging or from the consolidated
keys. Labelled exploratory. Adjacent-only merges could not reach the full count in 44 of 4,062
document runs because of the size cap (M-TP can merge non-adjacent chunks).

## 2026-10-13 09:50 — late chunking: out-of-memory fallback
The late-chunking arm (jina-embeddings-v3) stopped at Qasper paper 80 of 300 with an MPS
out-of-memory error: the plain encoder embeds 8 texts per batch, longest first, and a batch of
8,192-token chunk texts exceeded the 30 GB limit. A batch that runs out of memory is now
re-encoded one text at a time. Checked on 16 chunk texts: batch-of-1 and batch-of-8 embeddings
agree to cosine >= 0.99996 (bf16 rounding). Embeddings cached before the stop are reused; the
run resumes from the cache.
