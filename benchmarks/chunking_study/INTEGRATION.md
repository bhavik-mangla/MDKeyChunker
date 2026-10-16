# Integrating the v2 harness modules

New modules: `llm_api.py`, `answer.py`, `stats.py`; tests in `tests/test_harness_v2.py`
(`.venv/bin/python -m unittest tests/test_harness_v2.py -v`, no network).
`judge.py` is **not present yet**: writing it was blocked by the permission system in
this session. Its tests in `tests/test_harness_v2.py` are skipped until it exists.
The existing scripts are unchanged. The edits below are for the lead to make.

Housekeeping: add `results/spend_ledger.jsonl` to `.gitignore`. It is the live
spend record and must not be deleted while runs are in progress, because the budget
guard reads it.

## 1. build.py: route LLM calls through `llm_api`

1. Replace `OllamaJSON.call_json` and the body of `post_with_retry` with one call:
   ```python
   from llm_api import complete
   r = complete(self.spec, prompt, json_mode=True, max_tokens=max_tokens,
                temperature=0, seed=0, reasoning=self.reasoning, tag="enrich")
   self.in_tok += r["in_tokens"]; self.out_tok += r["out_tokens"]; self.cost += r["cost_usd"] or 0
   self.providers.add(r["provider"])
   return r["parsed_json"]          # None on unparseable output -> existing failure path
   ```
   `self.spec` replaces `self.model` and is a full spec, for example
   `ollama:qwen2.5:7b` or `openrouter:openai/gpt-oss-120b@deepinfra/bf16`.
   Add `cost_usd` and `providers` to `stats()`.
2. In `contextual_prefixes`, make the same swap (`json_mode=False, max_tokens=120`).
   Keep the document-first prompt so provider prefix caching works.
3. Add `--model-spec` (default `ollama:qwen2.5:7b`) and `--reasoning` (default unset;
   use `low` for gpt-oss). Write cached chunk sets under a spec-specific variant name
   (for example `enr_rk@gptoss`) so that strong-arm caches never overwrite 7B caches.
4. Retry policy from the plan: on `parsed_json is None`, retry up to 3 times, then keep
   the chunk with empty fields and count it. Do not drop it, and do not raise as
   `build_paper` does today for the strong arm.
5. Catch `llm_api.BudgetExceeded` in `run()` and stop the whole pool. Do not mark the
   paper as failed and continue.
6. Set `NUM_CTX` once per run. `llm_api` reads the same env var as `build.py`.

## 2. evaluate.py / evaluate_corpus.py: v2 statistics

1. In `summarize()`, replace `cluster_bootstrap(paired(...))` with:
   ```python
   import stats
   d = paired(pa, pb)
   b = stats.cluster_bootstrap(d, n=10_000, seed=0, strata=strata)   # strata=None for Qasper
   t = stats.tost(d, margin=MARGIN, boot=b)                          # reuses the bootstrap
   perm = stats.sign_flip_test(d, n=10_000)
   row.update(diff=b["mean"], ci95_bca=b["bca"][0.95], ci90_bca=b["bca"][0.90],
              ci95_pct=b["percentile"][0.95], verdict=t["verdict"], label=t["label"],
              p_tost=t["p_tost"], p_perm=perm["p"])
   ```
   Set `MARGIN = 0.05` for Qasper and `0.03` for FreshStack. Metrics are proportions,
   so ±5 and ±3 points become 0.05 and 0.03. For FreshStack, pass
   `strata={qid: topic}`.
2. After all comparisons, apply `stats.holm()` within each pre-registered family
   (ANALYSIS_PLAN_v2 §4–5):
   - TOST families use `p_tost`.
   - Superiority families (RQ1) use `p_perm`.
   - Store `p_holm` and `reject_holm` on each row.
3. Add `"verdict"` to `print_comparisons`.
4. Keep the old `cluster_bootstrap` for the v3 tables so they stay reproducible.

## 3. New script: end-to-end answers (`evaluate_e2e.py`, to write)

For each question and system, using the hybrid ranking already computed in evaluate:
```python
import answer
g = answer.generate_answer("qasper", q["question"], order, chunks, 512, GEN_SPEC)   # 1024 for freshstack
f1, typ = answer.qasper_answer_f1(g["answer"], answer.qasper_references(q_answers))
```
- `subset.json` has no gold answers. Load them from the parquet `qas.answers` for
  each qid. `qasper_references` accepts the parquet shape as well as the official
  JSON shape.
- Record `answer.prompt_hashes()` in the run metadata.
- Write per-question F1 in the existing `{cluster: [values]}` layout so that
  section 2 applies unchanged.
- FreshStack nugget recall and the Qasper LLM-judge score need `judge.py`. Its
  planned design:
  - FreshStack/AutoNuggetizer protocol: one call per answer covering all nuggets.
  - Labels `support` / `partial_support` / `no_support`, with strict parsing.
  - Metrics: A_strict, plus a partial-credit variant.
  - Controls: positive (gold answer), negative (seeded derangement), and length
    bias (`stats.spearman`).
  - A family guard against the generator and enrichers.
- The positive control needs the Stack Overflow `answer_text` from the HF queries
  parquet, which `data/freshstack_laravel.json` does not carry.

## 4. Smoke-test findings relevant to the plan

- **Generator: `openai/gpt-oss-120b@deepinfra/bf16`, reasoning low.** Served by
  DeepInfra. 205 tokens in, 33 out, of which 14 were reasoning tokens (about 40% of
  output). Cost $0.0000132.
- **Judge candidate: `meta-llama/llama-3.3-70b-instruct@deepinfra/turbo`.**
  - Served by DeepInfra. 109 tokens in, 44 out. Cost $0.0000250. Latency was 14 s.
  - The JSON verdict parsed cleanly.
  - It labelled two clearly supported nuggets as `partial_support`. Check that
    leniency in the positive control before accepting the judge.
- **Gemini free (`gemini-3.1-flash-lite`, key index 0).** One call, which succeeded.
