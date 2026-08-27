# Cross-section challenge workflow

The cross-section challenge is an exploratory analysis that tests whether
complete-paper contextual encoding becomes more useful when a question's
supporting evidence necessarily spans more than one top-level section.

It is separate from the frozen 200-paper primary experiment and does not alter
the primary QASPER test analysis.

## 1. Audit the prepared QASPER collection

After `sclc prepare`, run:

```bash
sclc qasper-audit --config configs/base.yaml
```

The audit examines every QASPER question with usable textual evidence in the
prepared collection. Two evidence-structure definitions are retained:

- **Strict cross-section required:** every acceptable evidence set spans more
  than one top-level section.
- **Minimal cross-section required:** every smallest acceptable evidence set
  spans more than one top-level section, although a larger alternative may not.

The dissertation challenge uses the **strict** definition and the original
QASPER **test split**.

The full audit identified:

- 302 strict candidates;
- 253 papers containing at least one strict candidate;
- 53 strict candidates in the original QASPER test split.

Audit outputs are written to:

```text
outputs/analysis/qasper_collection_audit/
```

## 2. Freeze and manually review the test candidates

Create the review files with:

```bash
sclc qasper-challenge --config configs/base.yaml --overwrite
```

The 53 strict test candidates were manually reviewed to verify that:

- evidence from more than one top-level section was genuinely required;
- the evidence-to-section mapping was valid;
- one section alone was not sufficient;
- duplicate evidence did not create an artificial cross-section label; and
- the annotated evidence supported the question.

The final review accepted **23 questions across 20 papers**:

- 18 questions from 15 cross-model papers;
- 5 questions from 5 Granite-extension papers.

The completed review decisions are committed at:

```text
data/subsets_cross_section_challenge/review_decisions.csv
```

The accepted Granite-extension questions remain part of the frozen manual-review
record, but they are not part of the final comparable challenge reported in the
dissertation.

## 3. Reproduce the reported challenge

The final dissertation reports:

- 18 accepted cross-model questions from 15 papers;
- Jina Embeddings v3;
- retrieval-unit sizes of 128, 256, and 512 tokens;
- section-isolated, section-constrained, and global contextual conditions;
- the same six principal retrieval metrics used in the primary experiment;
- 10,000 paired document-level bootstrap resamples.

The two signed scope comparisons are:

- section-isolated minus section-constrained;
- section-constrained minus global.

All signed differences follow **first condition minus second condition**. Holm
correction is applied across these two comparisons within each retrieval-unit
size and metric.

The convenience runner reproduces the reported Jina challenge:

```bash
./scripts/run_cross_section_challenge.sh \
  configs/cross_section_challenge.yaml \
  jina
```

Alternatively, after the three challenge conditions have been evaluated at each
size, run only the analysis stage:

```bash
sclc challenge-analyse \
  --model jina \
  --accepted-queries data/subsets_cross_section_challenge/review_decisions.csv \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/cross_section_challenge.yaml \
  --overwrite
```

Challenge-analysis outputs are written under:

```text
outputs/cross_section_challenge/analysis/cross_section_challenge_results/jina/
```

The main files are:

- `summary_by_retrieval_unit_size.csv`;
- `comparisons_within_retrieval_unit_size.csv`;
- `query_scope_effects.csv`;
- `manifest.json`;
- bootstrap arrays under `bootstrap/`.

The challenge is deliberately small and selected to stress contextual encoding
scope. Its results should therefore be interpreted as exploratory supporting
evidence rather than as an additional representative test set.
