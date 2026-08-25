# Full QASPER evidence-structure audit

Run after `sclc prepare`:

```bash
sclc qasper-audit --config configs/base.yaml
```

The command audits every usable textual-evidence question in the prepared QASPER
train, validation, and test splits. It does not alter the frozen experimental
sample and does not run tokenisation, encoding, retrieval, or evaluation.

Two cross-section definitions are retained:

- **Strict cross-section required:** every acceptable evidence set spans more
  than one top-level section.
- **Minimal cross-section required:** every smallest acceptable evidence set
  spans more than one top-level section, although a larger alternative may not.

Only the strict definition should be used to construct a cross-section challenge
set. Validation candidates are flagged because validation informed chunk-size
selection. Train and test candidates are reported separately and no candidate is
selected automatically.

Outputs are written to `outputs/analysis/qasper_collection_audit/`.
