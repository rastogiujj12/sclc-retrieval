# Chunk-size sensitivity analysis

Version 0.7.8 extends the dense scope ablation from the selected 128-token
configuration to 128, 256, and 512 canonical tokens.

The sensitivity experiment does not change the frozen paper sample or question
set. For each embedding model, every eligible question associated with the
selected papers is evaluated under:

- fixed-size dense chunking;
- section-isolated chunking;
- section-constrained late chunking;
- global late chunking.

The analysis reports both:

- `all_questions`: every model-eligible question associated with the frozen
  selected-paper corpus;
- `split_test`: the original held-out QASPER test subset.

For Granite, results are also separated into the cross-model core, the
Granite-only extension, and all eligible papers. Jina is restricted to the
cross-model core by the existing no-truncation eligibility rule.

## Run the full pipeline

Run one model at a time so Colab can resume safely from cached outputs:

```bash
./scripts/run_chunk_size_sensitivity.sh configs/base.yaml granite
./scripts/run_chunk_size_sensitivity.sh configs/base.yaml jina
```

The script skips compatible cached outputs. It constructs retrieval units and
runs encoding, retrieval, and evaluation for all four dense conditions at 128,
256, and 512 tokens. It then runs the aggregate sensitivity analysis.

## Run only the analysis

After all required evaluations exist:

```bash
sclc chunk-size-sensitivity \
  --model granite \
  --chunk-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

Repeat with `--model jina` after the Jina evaluations are complete.

## Outputs

Outputs are written to:

```text
outputs/analysis/chunk_size_sensitivity/<model>/
```

The main files are:

- `summary_by_chunk_size.csv`: mean metrics for each condition, chunk size,
  analysis set, and sample scope;
- `comparisons_within_chunk_size.csv`: document-level paired bootstrap
  comparisons between adjacent dense conditions at each chunk size;
- `query_scope_effects.csv`: per-query isolated-minus-section and
  section-minus-global effects;
- `scope_interactions_across_chunk_sizes.csv`: direct tests of whether those
  scope effects change between 128, 256, and 512 tokens;
- `manifest.json`: full configuration and source fingerprints.

Fixed-rank metrics are retained because they describe early-ranking behaviour,
but they retrieve different quantities of text at different chunk sizes. The
fixed-token-budget metrics therefore provide the fairer cross-size comparison.
