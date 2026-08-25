# Evidence-Structure Analysis

This secondary analysis addresses the low number of manually coded multi-hop
questions without changing the original QASPER sample or its confirmatory
overall result.

## Classification unit

The unit is an acceptable QASPER evidence set. Paragraphs are mapped to the
same top-level academic section boundaries used by section-constrained late
chunking.

Each acceptable set is classified as:

- `single_paragraph`: one distinct paragraph;
- `multi_paragraph_same_section`: multiple distinct paragraphs within one
  top-level section;
- `multi_paragraph_cross_section`: multiple distinct paragraphs spanning more
  than one top-level section.

## Alternative acceptable sets

QASPER can provide several acceptable support sets for one question. The
primary label is chosen as follows:

1. retain the set or sets containing the fewest distinct paragraphs;
2. if tied sets have different structures, choose the least-distributed
   structure (`single` before `same-section` before `cross-section`);
3. preserve the tie in `minimal_structure_variants` and
   `minimal_structure_ambiguous`;
4. report `cross_section_possible` and `cross_section_required` separately.

This rule matches the existing complete-support metric, under which retrieving
all paragraphs from any acceptable set counts as success. It also prevents the
union of alternative annotations from falsely making a question look
cross-sectional.

## Statistical scope

The primary stratified comparisons use only conditions with identical
section-bounded target spans:

- section-isolated vs section-constrained;
- section-constrained vs global;
- section-isolated vs global.

The command uses the configured confirmatory split, metrics, paired
document-level bootstrap, confidence level, seed, and Holm adjustment.
Low-count strata are flagged and remain secondary exploratory findings.

## Command

```bash
sclc evidence-structure \
  --model granite \
  --chunk-size 128 \
  --config configs/base.yaml
```

No encoding, retrieval, or evaluation stage is repeated.
