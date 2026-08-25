# QASPER strict cross-section challenge freeze

The primary 180-question experiment remains unchanged. This supplementary
challenge set is frozen before any new challenge retrieval is run.

Run the complete collection audit first:

```bash
sclc qasper-audit --config configs/base.yaml
```

Then freeze the challenge:

```bash
sclc qasper-challenge --config configs/base.yaml --overwrite
```

The command selects only official QASPER **test-split** questions for which
every acceptable evidence set spans multiple top-level sections. Validation is
excluded because it informed chunk-size selection. No retrieval scores are used
in selection or review ordering.

Frozen groups:

- `cross_model_complete.csv`: all 42 Jina- and Granite-compatible strict test
  questions.
- `cross_model_unseen.csv`: the 37 cross-model questions absent from the primary
  experiment.
- `granite_extension.csv`: 11 strict test questions from longer Granite-only
  documents.

Before new retrieval, complete `blinded_review_sheet.csv` without opening
`review_key.csv` or condition-level outputs. Reviewers decide whether multiple
sections are genuinely required, whether support is duplicated, whether the
section mapping is valid, and whether the evidence supports the question.

The freeze also writes source and output SHA-256 hashes plus a configuration
fingerprint to `manifest.json`.
