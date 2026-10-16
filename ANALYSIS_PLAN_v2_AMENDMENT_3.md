# Amendment 3 to analysis plan v2: related keys and single-call extraction (H5, H6)

Date: 2026-10-08. Written before any outcome involving key-linked expansion,
per-field extraction, or key/entity indexing was computed. At this time the
H4 merge arms (Amendment 2) are being scored; no H5/H6 arm exists. Plan v2,
Amendment 1 and Amendment 2 are unchanged. Both families below are
**pre-specified secondary** (written after H1-H3 were seen).

Two parts of the system were generated but never tested: the related keys
returned by every enrichment call, and the choice of one call for all fields
instead of one call per field.

## A. H5: related keys at retrieval time (no LLM calls)
Arms, on the cached E chunks (local model), hybrid ranking as in plan v2:
- **E+KX (key expansion)**: walk the ranked list; after each retrieved chunk,
  insert the not-yet-included chunks of the same source document that are
  key-linked to it (same key; key in its related keys; or its key in their
  related keys), in document order. The expanded order is then scored with
  the plan v2 budgeted metrics (the budget cuts it as usual).
- **E+AX (adjacency expansion, control)**: the same procedure, but each
  retrieved chunk brings the same number of chunks as E+KX would add for it,
  taken from its nearest neighbours in the same document (next, previous,
  next+1, ...), skipping chunks already included.
Tests (hybrid; plan v2 primary metric per dataset; Holm over the 4 tests,
two-sided sign-flip p):
- **H5a**: E+KX - E. **H5b**: E+KX - E+AX.
Each on both datasets. Reported with the share of retrieved chunks that have
at least one key link.

## B. H6: one call for all fields vs one call per field (Qasper)
- **PF (per-field)**: four separate calls per chunk, one each for title,
  summary, keywords and questions (the four indexed fields), each using that
  field's instruction from ENRICH_PROMPT and the same context (section path,
  position, previous chunk's summary from PF's own summary call). Same local
  model, options and exclusion rules as E. No key or rolling dictionary
  (keys are not indexed).
- Sample: 100 of the 300 Qasper v2 papers, `random.Random(2027).sample` of the
  sorted paper ids. FreshStack is excluded (compute).
- **H6**: E - PF on rec@512t, hybrid, TOST at +/-5 (single test, alpha 0.05).
  Also reported: input/output tokens and wall time per chunk for E and PF.

## C. Exploratory (unadjusted, labelled)
- Index keys and entities too (render "meta+key"): E(meta+key) - E, both
  datasets, all retrievers. No new LLM calls.
- H5 and H6 under bm25, nomic and mxbai; FreshStack per domain for H5.
