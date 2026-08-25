# Changelog

## 0.7.10

- Added interruption-safe dense passage caching through a fingerprinted partial manifest.
- Completed per-document passage embeddings now survive a runtime interruption and are validated and reused on restart.
- Final and partial manifests are written atomically, and forced overwrites remove stale state before rebuilding.
- Added regression coverage for resuming a two-document global encoding after a synthetic interruption; the suite now contains 49 tests.

## 0.7.8

- Added a full dense chunk-size sensitivity workflow for 128, 256, and 512 token units.
- Added `scripts/run_chunk_size_sensitivity.sh`, which runs all four dense conditions for one model while safely reusing compatible cached outputs.
- Added `sclc chunk-size-sensitivity` to summarise all eligible selected-paper questions and the original held-out test subset.
- Added document-level paired bootstrap comparisons within each chunk size and direct chunk-size-by-scope interaction tests.
- Added separate Granite core, Granite extension, and all-eligible analyses while preserving Jina's cross-model eligibility restriction.
- Added regression coverage; the suite now contains 46 tests.

## 0.7.7

- Added `sclc qasper-challenge` to freeze the strict cross-section test challenge before new retrieval runs.
- Produces the 42-question cross-model set, its 37-question previously unseen subset, and the separate 11-question Granite extension.
- Added deterministic blinded review IDs, a review sheet, a separate review key, fixed inclusion criteria, SHA-256 checksums, and a frozen challenge fingerprint.
- Preserves the original 180-question primary experiment unchanged.
- Added regression tests for challenge counts, subset membership, and review blinding.

## 0.7.5

- Add `sclc evidence-structure` as a secondary analysis that leaves the original
  QASPER test distribution unchanged.
- Classify each query as `single_paragraph`,
  `multi_paragraph_same_section`, or `multi_paragraph_cross_section` using the
  least-distributed minimal acceptable QASPER evidence set.
- Preserve alternative-set ambiguity, cross-section possibility, and whether
  cross-section support is strictly required.
- Add descriptive counts, a query-type cross-tab, long-form metric summaries,
  and document-level paired bootstrap scope comparisons with Holm correction.
- Restrict inferential scope comparisons to identical section-bounded target
  spans: isolated vs section-constrained, section-constrained vs global, and
  isolated vs global.
- Add low-sample warnings instead of treating small evidence strata as
  confirmatory.
- Expand the test suite to 43 tests.

## 0.7.4

- Add `--model` to `sclc compare` so Granite confirmatory statistics can be run
  before the optional Jina robustness experiment.
- Add the same model filter to `sclc analyse`.
- Namespace model-specific comparison and scope-effect outputs to prevent partial
  runs from overwriting future all-model results.
- Keep the previous all-model behaviour when `--model` is omitted.


## 0.7.3

- Add `flex_attention` as a validated Transformers attention backend.
- Include FlexAttention in dense-cache fingerprints through the existing backend field.
- Update CUDA OOM guidance for long-context Turing GPU runs.


## 0.7.2

- Force PyTorch SDPA by default for memory-efficient long-context inference.
- Add configurable `dense.attn_implementation` with strict validation.
- Include the attention backend in passage and query cache fingerprints.
- Replace raw CUDA OOM failures with an actionable message explaining why batch-size
  reduction cannot repair contextual-scope OOMs.
- Document the larger-GPU fallback without permitting truncation or windowing.

## 0.7.1

- Consolidated the wide evaluation summary DataFrame before `reset_index()` to
  prevent pandas fragmentation warnings without changing any metric values.
- Added a regression check that treats `pandas.errors.PerformanceWarning` as an
  error during evaluation tests.

## 0.7.0

- Added a validation-only chunk-size pilot for 128, 256, and 512 token units.
- Namespaced every chunk-dependent artefact under `chunk_<size>` so pilot runs
  cannot overwrite one another.
- Added `--chunk-size` to chunking, encoding, retrieval, evaluation, comparison,
  and scope-effect analysis commands.
- Added `sclc select-chunk-size`, which selects a common chunk size using Granite
  fixed-dense validation performance and a predeclared practical-equivalence rule.
- Added complete within-paper rankings.
- Added fixed-rank metrics at 1, 3, 5, and 10, including nDCG, Recall, available-
  depth Precision, strict Precision, evidence-paragraph recall, evidence-span
  coverage, and complete-evidence recovery.
- Added fixed-token-budget evidence metrics at 512, 1,024, 2,048, and 4,096
  retrieved tokens.
- Added MAP, reciprocal rank, R-Precision, full-ranking nDCG, first-relevant rank,
  normalised first-relevant rank, and rank percentile.
- Added candidate-count, retrieved-fraction, and cutoff-saturation diagnostics.
- Restricted chunk-size selection to the validation split and confirmatory
  comparisons to the test split.
- Added document-level paired bootstrap comparisons between pilot chunk sizes.
- Added pilot and selected-size shell scripts.
- Expanded the test suite to 39 tests.

## 0.6.0

- Made the union of distinct QASPER evidence paragraphs the primary relevance
  policy and retained best-evidence-set scores as sensitivity analysis.
- Corrected non-whitespace evidence overlap and coverage handling.
- Required blind query-type coding before retrieval and accepted `uncertain`.
- Added structural scope-effect analysis and stronger cache fingerprints.
- Updated model loading to use the current `dtype` argument.

## 0.5.0

- Added exact within-document BM25 and dense retrieval.
- Added deterministic tie-breaking and fingerprinted ranking caches.
- Added retrieval evaluation, query-type summaries, paired bootstrap confidence
  intervals, Holm correction, and error-analysis outputs.

## 0.4.0

- Added all four dense representation conditions and shared query encoding.
- Standardised target-token mean pooling and L2 normalisation.
- Added strict representation fingerprints and resumable per-document encoding.

## 0.7.6

- Added `sclc qasper-audit` to inspect evidence structure across the complete
  prepared QASPER collection without changing the frozen primary sample.
- Added strict and minimal cross-section definitions, current-sample overlap,
  model-length eligibility, and split-aware candidate statuses.
- Added auditable query-, candidate-, and document-level CSV outputs for a
  possible supplementary cross-section challenge set.

## v0.7.9 — Cross-section challenge completion

- Added validation and freezing of completed blinded review decisions.
- Added generation of an accepted-document manifest for a separate challenge run.
- Added accepted-set summaries and document-level bootstrap comparisons for
  section-isolated, section-constrained, and global retrieval across chunk sizes.
- Added tests for review finalisation and challenge-specific analysis.
