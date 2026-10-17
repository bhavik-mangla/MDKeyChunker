# Golden documents: sources and licences

`docs/` holds real documents, unmodified except where an excerpt is noted. `expected/` holds
the `mk.chunk()` output for each one (`Chunk.to_dict()`), regenerated with
`MDK_UPDATE_GOLDEN=1 pytest tests/golden`. Review snapshot diffs in PRs like code.

| File | What it exercises | Source | Licence |
|---|---|---|---|
| 01-readme.md | nested lists, tables, badges | this repository, `README.md` at `a279b0b` | MIT |
| 02-analysis-plan-v2.md | long sections, tables | this repository, `benchmarks/chunking_study/ANALYSIS_PLAN_v2.md` | MIT |
| 03-laravel-database-testing.md | code-heavy docs (FreshStack corpus of the v4 study) | laravel/docs `724c31ccd3edce6b6dfe5e0dd2a594a47217f078`, `database-testing.md` | MIT |
| 04-laravel-contracts.md | an 80-row table (FreshStack corpus) | laravel/docs `724c31cc...`, `contracts.md` | MIT |
| 05-angular-change-detection.md | tables, HTML-ish components (FreshStack corpus) | angular/angular `7d9b38e97bbef0c6f6e3761e7cd86457c1e0f2e1`, `adev/src/content/guide/pipes/change-detection.md` | MIT |
| 06-angular-routing-urlmatcher.md | code fences indented inside list items (list content, not top-level fences) | angular/angular `7d9b38e9...`, `adev/src/content/guide/routing/routing-with-urlmatcher.md` | MIT |
| 07-qasper-1803.02914.md | an arXiv paper as Markdown (Qasper corpus of the v4 study) | Qasper (Dasigi et al., 2021), paper 1803.02914 | CC BY 4.0 |
| 08-trpl-zh-cn-installation.md | CJK text | KaiserY/trpl-zh-cn `4d904df5f578d7938a6cc58c265babec0ed49421`, `src/ch01-01-installation.md` | MIT |
| 09-github-docs-basic-formatting.md | YAML front matter, Liquid tags, nested lists, tables | github/docs `6ed0ad8e50adda37a6b596bda8789788e28a0c0d`, `content/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax.md` | CC BY 4.0 |
| 10-commonmark-spec-code-blocks.md | fence edge cases (long fences, `~~~`, nesting); front matter ended by `...` | commonmark/commonmark-spec `8a5f7646e9f67c3004f023d262b3e6b596afc2ed`, `spec.txt`: front matter + "Indented code blocks" and "Fenced code blocks" sections (excerpt) | CC BY-SA 4.0 |
| 11-alice-chapter-1.txt | headerless plain text with CRLF (window fallback) | Project Gutenberg eBook #11, *Alice's Adventures in Wonderland*, Chapter I (excerpt, Gutenberg header/footer removed) | Public domain |

The CommonMark examples used by `tests/property/test_markdown_it_differential.py` are in
`tests/property/data/` (CommonMark Spec 0.31.2, CC BY-SA 4.0).
