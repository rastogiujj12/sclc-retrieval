# Chunk-Size Pilot Implementation — v0.7.0

## Purpose

The original 512-token design causes Recall@10 saturation for many short
within-paper candidate sets. Version 0.7.0 adds a pre-confirmatory pilot at 128,
256, and 512 tokens using BM25 and Granite fixed dense retrieval.

## Methodological safeguards

- Chunk size is selected on QASPER validation questions only.
- Test questions are excluded from selection.
- All three sizes use the same selected papers, questions, evidence policy,
  tokenizer, zero overlap, and retrieval-within-source-paper rule.
- Fixed-token-budget evidence metrics complement fixed-rank metrics because a
  fixed number of 128-token chunks contains less text than the same number of
  512-token chunks.
- One selected size is frozen before section-isolated, section-constrained, and
  global confirmatory comparisons.
- Confirmatory comparison and scope-effect commands use the test split only.

## Selection rule

The selected size is based on Granite fixed dense validation performance:

1. Highest evidence-paragraph recall under a 1,024-token budget.
2. Treat differences within 0.01 as practically equivalent.
3. Among primary-equivalent sizes, use complete acceptable evidence-set
   recovery under a 2,048-token budget.
4. Again treat differences within 0.01 as practically equivalent.
5. Choose the largest remaining size as an efficiency tie-break.

BM25, MAP, nDCG@5, Recall@5, and paired-bootstrap intervals are retained as
robustness evidence.

## New outputs

- `data/retrieval_units/chunk_<size>/`
- `outputs/encodings/chunk_<size>/`
- `outputs/rankings/chunk_<size>/`
- `outputs/evaluation/chunk_<size>/`
- `outputs/analysis/chunk_size_pilot/`

The global blind coding file remains
`data/retrieval_units/query_types.csv`.

## Added metrics

- nDCG, Recall, available-depth Precision, strict Precision at 1/3/5/10
- evidence-paragraph recall and evidence-span coverage at 1/3/5/10
- complete acceptable-set and complete-union recovery at 1/3/5/10
- MAP, MRR, R-Precision, full-ranking nDCG
- first relevant rank, normalised rank, and rank percentile
- fixed-token-budget evidence metrics at 512/1024/2048/4096 tokens
- candidate-count, retrieved-fraction, and saturation diagnostics

## Validation status

- Python bytecode compilation passed.
- 39 automated tests passed.
- Shell scripts passed `bash -n` validation.
- CLI help and the new command wiring were exercised with import stubs because
  the packaging environment did not contain the optional runtime downloads.
- Ruff and mypy were not available in the packaging environment and should be
  run locally after `pip install -e ".[dev]"`.
