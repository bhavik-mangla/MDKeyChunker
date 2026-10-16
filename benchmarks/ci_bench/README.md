# CI benchmark: tiny Qasper subset

A regression check, not evidence. It checks that evidence matching, the token budget and
BM25 still give the numbers they gave when this subset was frozen. It says nothing about
which chunker is better. With n = 39 questions, differences between arms are mostly noise.
For example, here S+T scores below S (0.372 vs 0.415, cl100k), whereas on all 951 Qasper
questions the study found no detectable S+T - S difference under BM25 (+0.2 points, 95% CI
[-0.6, +0.9]; exploratory). The test asserts no ordering between arms, only the frozen values.

- `qasper_ci.json`: 10 papers (39 questions with gold evidence) drawn with seed 2026 from the
  v4 study's 300 Qasper papers. The pool is the 179 papers with at least 3 such questions,
  under 40,000 characters of Markdown, and both cached chunk sets. Each paper keeps its
  Markdown, its questions and evidence, and the study's frozen `struct` and `fixed512` chunks.
  Built by `make_subset.py` from the study's `data/v2`.
- `run.py`: BM25 (`method="study"`, bit-identical to the study's rank-bm25) within each paper,
  recall of gold evidence within 512 tokens, for three arms: `F512c` (512-character windows),
  `S` (header-led chunks) and `S+T` (header-led chunks with the section-path prefix).
- `golden.json`: frozen results for the zero-dependency tokenizer (`core`) and for tiktoken
  `cl100k_base`. With `cl100k_base`, all 117 per-question values (39 questions x 3 arms)
  equal the study's `results/v2/qasper_per_question.json`.

`tests/test_ci_bench.py` asserts the means within +/-1 point and every per-question value
unchanged. To refreeze after an intended change: `uv run python benchmarks/ci_bench/run.py
--tokenizer core --write` (and `--tokenizer cl100k_base --write` with tiktoken installed), and
say why in the PR.

## Data licence and attribution

Qasper: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner.
"A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers."
NAACL 2021. Released by the Allen Institute for AI under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The papers are converted to
Markdown, and the questions are filtered to those with text evidence, as described in the
study's `benchmarks/chunking_study/data/v2/DATASETS_V2.md`. No other changes were made.
