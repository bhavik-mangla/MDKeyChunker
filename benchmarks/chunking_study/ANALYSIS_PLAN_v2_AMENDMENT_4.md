# Amendment 4 to analysis plan v2: post-hoc key consolidation and section expansion

Date: 2026-10-08. Written before E-noRK (Amendment 2) is built and before any
H5 (Amendment 3) outcome is computed. Earlier plan text is unchanged. Added
after a literature check: the closest prior mechanism to the rolling key
dictionary is TopicGPT-style label reuse (Pham et al., NAACL 2024), where
labels are assigned and near-duplicates are merged in a separate pass. A
reviewer can ask whether the in-call dictionary beats that cheaper pattern.

## A. H7: in-call rolling dictionary vs post-hoc consolidation (pre-specified secondary)
- **M-TP**: start from the E-noRK chunks (keys assigned without the
  dictionary). One extra call per document (same local model, temperature 0,
  seed 0) receives the document's distinct keys in order of first use and
  returns a JSON map from each key to a canonical key, merging keys that name
  the same specific aspect; the key rules from ENRICH_PROMPT are included
  verbatim. Keys are remapped (unmapped or invalid -> unchanged) and Stage 3 is
  applied. A failed call leaves the keys unchanged (counted).
- **H7**: M - M-TP on both datasets (hybrid; plan v2 primary metric; two-sided
  sign-flip; Holm over the 2 tests). Reported with key reuse rate, distinct
  keys per document, Stage 3 merges, and LLM calls/tokens for E (dictionary)
  vs E-noRK + consolidation.

## B. Exploratory addition to H5 (unadjusted, labelled)
- **E+SX (section expansion)**: after each retrieved chunk, insert the
  not-yet-included chunks of the same document with the same section path, in
  document order (parent-section retrieval). Not count-matched. Compared with
  E and E+KX under the plan v2 budgets.
