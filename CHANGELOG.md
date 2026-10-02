# Changelog

## 0.3.0 — 2026-10-02

### Fixed
- **LLM enrichment works again.** `ENRICH_PROMPT` had been replaced by a truncated stub in 0.2.0's June commits, so the chunk text never reached the model and LLM mode returned no keys or summaries.
- One failed LLM call no longer aborts the document. Authentication errors and unknown providers fail fast.
- Malformed LLM output is coerced to the expected types. `related_keys` are lowercased and limited to keys from the rolling dictionary.
- **Chunker:** headers now start a chunk and always stay with their content (previously about a third of chunks on header-dense documents ended on a dangling header). Small chunks are merged forward, and the larger part names the section.
- **Chunker:** code fences follow CommonMark (fence length and character, at most 3 spaces of indent, no backticks in a backtick info string). Setext headers, CRLF line endings, `1)` lists and trailing `#`s are handled.

### Changed
- Chunk boundaries differ from 0.2.x on most documents. Re-index after upgrading.
- `spacy` and `anthropic` are optional extras (`pip install "mdkeychunker[spacy]"`, `"mdkeychunker[anthropic]"`).
- The SciFact benchmark moved out of the installed package into `benchmarks/`.

### Added
- `CITATION.cff`, a citation section in the README, and CI on Python 3.10–3.13.

## 0.2.0
- Single-call enrichment with rolling keys and key-based restructuring (the version described in arXiv:2603.23533 v1/v2, commit `e3e1b86`).
