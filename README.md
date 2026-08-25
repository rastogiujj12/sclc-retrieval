# Section-Constrained Late Chunking Retrieval Experiment

Version **0.7.8** implements the experimental artefact for the dissertation:

> **Section-Constrained Late Chunking for Academic Retrieval: An Ablation Study**

The project tests whether the amount of document context available during
embedding formation should respect academic section boundaries.

## Experimental conditions

| Condition | Retrieval units | Encoding scope |
|---|---|---|
| `bm25` | Continuous fixed-token chunks | Lexical baseline |
| `fixed_dense` | Continuous fixed-token chunks | Target chunk only |
| `section_isolated` | Section-bounded chunks | Target chunk only |
| `section_constrained` | Same section-bounded chunks | Parent top-level section |
| `global` | Same section-bounded chunks | Complete paper |

The three contextual-scope conditions use identical target spans. Their only
intended difference is whether the target representation is influenced by the
chunk, its parent section, or the complete paper.

## Why version 0.7.0 includes a chunk-size pilot

QASPER retrieval is restricted to the question's source paper. With 512-token
chunks, many short papers contain ten or fewer candidates, making Recall@10
saturated or nearly saturated. Version 0.7.0 therefore compares **128, 256, and
512 token chunks** before the confirmatory contextual-scope experiment.

The pilot uses:

- BM25;
- Granite fixed-size dense retrieval;
- QASPER's **validation split only**;
- fixed-rank metrics at 1, 3, 5, and 10;
- fixed retrieved-token budgets of 512, 1,024, 2,048, and 4,096 tokens;
- complete within-paper rankings;
- document-level paired bootstrap confidence intervals.

One common chunk size is then frozen for the remaining methods. The test split
is not used for chunk-size selection.

## Requirements

- Python 3.11 or 3.12
- A virtual environment is recommended
- Internet access for the first Hugging Face download
- A CUDA GPU is recommended for dense encoding, but CPU execution is supported

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

The included suite currently contains **46 tests**.

## Configuration

The default configuration is `configs/base.yaml`.

Important fixed controls include:

```yaml
chunking:
  supported_chunk_sizes: [128, 256, 512]
  overlap_tokens: 0
  retain_short_final_chunk: true

evaluation:
  cutoffs: [1, 3, 5, 10]
  token_budgets: [512, 1024, 2048, 4096]
  primary_metric: ndcg_at_5
  confirmatory_split: test

pilot:
  chunk_sizes: [128, 256, 512]
  selection_split: validation
  primary_metric: evidence_paragraph_recall_at_token_budget_1024
  secondary_metric: complete_evidence_at_token_budget_2048
  practical_equivalence_margin: 0.01
```

Before the final dissertation run, pin both embedding models and the canonical
tokenizer to immutable Hugging Face commit revisions.

## Data preparation already used by this project

The deterministic dataset stages are:

```bash
sclc prepare --config configs/base.yaml
sclc profile --config configs/base.yaml
sclc sample --config configs/base.yaml
```

They produce reconstructed QASPER papers, complete-paper token profiles, and a
reproducible paper-level sample.

The selected corpus is designed as:

- 150 `cross_model_core` papers that fit Granite and Jina;
- 50 `granite_extended` papers that fit Granite but exceed Jina's limit.

References are removed, complete papers are never windowed for global late
chunking, and longer-than-Granite papers are excluded.

## Query-type coding

Query categories must be assigned before retrieval results are inspected. The
single global coding file is:

```text
data/retrieval_units/query_types.csv
```

Allowed labels are:

```text
factual
section_specific
multi_hop
synthesis
uncertain
```

The coding is independent of chunk size, so it is not duplicated under the
128/256/512 namespaces.

## Chunk-size pilot workflow

### Option A: run the provided script

After `query_types.csv` exists:

```bash
./scripts/run_chunk_size_pilot.sh configs/base.yaml
```

The script constructs and evaluates all three chunk sizes for BM25 and Granite
fixed dense retrieval, then runs validation-only selection.

### Option B: run each stage manually

For each size:

```bash
sclc chunk --chunk-size 128 --config configs/base.yaml
sclc encode --condition bm25 --chunk-size 128 --config configs/base.yaml
sclc retrieve --condition bm25 --chunk-size 128 --config configs/base.yaml
sclc evaluate --condition bm25 --chunk-size 128 --config configs/base.yaml
```

Repeat with `--chunk-size 256` and `--chunk-size 512`.

Then run Granite fixed dense retrieval for each size:

```bash
sclc encode \
  --condition fixed_dense \
  --model granite \
  --chunk-size 128 \
  --config configs/base.yaml

sclc retrieve \
  --condition fixed_dense \
  --model granite \
  --chunk-size 128 \
  --config configs/base.yaml

sclc evaluate \
  --condition fixed_dense \
  --model granite \
  --chunk-size 128 \
  --config configs/base.yaml
```

Again repeat for 256 and 512.

Finally select the common chunk size:

```bash
sclc select-chunk-size --config configs/base.yaml
```

The selection result is written to:

```text
outputs/analysis/chunk_size_pilot/selection.json
```

### Selection rule

The method is deliberately fixed before viewing the pilot results:

1. Use Granite fixed-dense results on the validation split.
2. Find the best mean evidence-paragraph recall under a 1,024-token retrieval
   budget.
3. Retain sizes within 0.01 of that best mean.
4. Among them, retain sizes within 0.01 of the best complete-evidence recovery
   under a 2,048-token budget.
5. If more than one size remains, choose the largest for storage and computation
   efficiency.

BM25 and the additional ranking metrics are reported as robustness evidence but
do not override the predeclared Granite selection rule.

## Full dense chunk-size sensitivity

After the 128-token primary run, version 0.7.8 can evaluate all four dense
conditions at 128, 256, and 512 tokens for every eligible question associated
with the frozen selected-paper corpus. Run one model at a time:

```bash
./scripts/run_chunk_size_sensitivity.sh configs/base.yaml granite
./scripts/run_chunk_size_sensitivity.sh configs/base.yaml jina
```

The script reuses compatible cached results, so an existing 128-token run is not
recomputed. It finishes by running:

```bash
sclc chunk-size-sensitivity \
  --model granite \
  --chunk-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

The analysis reports all eligible selected-paper questions and the original test
subset separately. It also directly tests whether the
section-constrained-minus-global effect changes with chunk size. Results are
written to:

```text
outputs/analysis/chunk_size_sensitivity/<model>/
```

See `docs/chunk_size_sensitivity.md` for output definitions and interpretation
rules. Fixed-token-budget metrics should be prioritised for cross-size
comparisons because fixed-rank cutoffs retrieve more text as chunks become
larger.


## Evidence-structure analysis

The natural QASPER distribution contains many factual questions and relatively
few manually coded multi-hop questions. The secondary evidence-structure
analysis does not rebalance, oversample, or relabel the held-out test set. It
instead groups each query by the location of its acceptable gold evidence:

- `single_paragraph`;
- `multi_paragraph_same_section`;
- `multi_paragraph_cross_section`.

QASPER may contain multiple alternative acceptable evidence sets. The primary
label uses the complete acceptable set with the fewest distinct paragraphs. If
several minimal sets are tied, the least-distributed structure is used because
complete-support evaluation succeeds when any acceptable set is recovered; the
tie and all alternative structures remain recorded for audit. Consequently, a
query is labelled cross-section only when the minimal complete support route
still crosses top-level academic sections.

After the selected-size evaluations exist, run Granite independently:

```bash
sclc evidence-structure \
  --model granite \
  --chunk-size 128 \
  --config configs/base.yaml
```

The command requires no re-encoding, retrieval, or evaluation. It writes to:

```text
outputs/analysis/chunk_128/evidence_structure/granite/
```

Important outputs are:

```text
evidence_structure_queries.csv
evidence_structure_counts.csv
evidence_structure_query_type_crosstab.csv
summary_by_evidence_structure.csv
scope_comparisons_by_evidence_structure.csv
manifest.json
```

The inferential analysis is deliberately limited to the three conditions with
identical section-bounded target spans: section-isolated,
section-constrained, and global. Document-level paired bootstrap intervals and
Holm correction are calculated separately within each evidence structure. Rows
with fewer than 20 queries or 10 papers are explicitly flagged as low-sample
secondary findings. Evidence distribution should not be described as proof of
reasoning depth: multiple paragraphs can be redundant or alternative rather
than genuinely multi-hop.


## Metrics

### Fixed-rank metrics

At `k = 1, 3, 5, 10`, the evaluator reports:

- nDCG@k;
- Recall@k;
- available-depth Precision@k, whose denominator is the number returned when a
  paper has fewer than k candidates;
- strict Precision@k, whose denominator is always k;
- unique evidence-paragraph recall;
- non-whitespace evidence-span coverage;
- recovery of at least one complete acceptable QASPER evidence set;
- stricter recovery of the complete evidence union;
- best-acceptable-set sensitivity scores.

### Complete-ranking metrics

Because candidate ranking is restricted to one source paper, complete rankings
are stored and used to calculate:

- Mean Average Precision at query level (`average_precision`);
- reciprocal rank;
- R-Precision;
- full-ranking nDCG;
- first relevant rank;
- normalised first relevant rank;
- first relevant rank percentile.

### Fixed-token-budget metrics

For 512, 1,024, 2,048, and 4,096 retrieved tokens, the evaluator reports:

- evidence-paragraph recall;
- evidence-span coverage;
- complete acceptable evidence-set recovery;
- complete-union recovery;
- number of chunks and actual tokens retrieved.

These metrics make 128-, 256-, and 512-token retrieval more comparable than a
fixed number of chunks alone.

### Saturation diagnostics

Each query records:

- candidate count;
- whether each rank cutoff retrieves the complete candidate set;
- retrieved fraction at each cutoff.

Recall@10 can therefore be reported separately for saturated and unsaturated
queries rather than interpreted as equally informative everywhere.

## Output namespaces

Chunk-dependent artefacts cannot overwrite another chunk-size run:

```text
data/retrieval_units/
├── query_types.csv
├── query_type_coding.csv
├── chunk_128/
├── chunk_256/
└── chunk_512/

outputs/
├── encodings/
│   ├── queries/                  # shared query embeddings
│   ├── chunk_128/
│   ├── chunk_256/
│   └── chunk_512/
├── rankings/
│   ├── chunk_128/
│   ├── chunk_256/
│   └── chunk_512/
├── evaluation/
│   ├── chunk_128/
│   ├── chunk_256/
│   └── chunk_512/
└── analysis/
    ├── chunk_size_pilot/
    └── chunk_<selected_size>/
```

Every stage uses fingerprints and refuses incompatible cached artefacts unless
`--overwrite` is supplied.

## After selecting a chunk size

Suppose `selection.json` chooses 256. Encode the remaining Granite conditions:

```bash
for condition in section_isolated section_constrained global; do
  sclc encode \
    --condition "$condition" \
    --model granite \
    --chunk-size 256 \
    --config configs/base.yaml
done
```

The fixed-dense Granite encoding from the pilot is reused.

Then encode the Jina conditions on the cross-model core:

```bash
for condition in fixed_dense section_isolated section_constrained global; do
  sclc encode \
    --condition "$condition" \
    --model jina \
    --chunk-size 256 \
    --config configs/base.yaml
done
```

Run retrieval and evaluation for the selected size with:

```bash
./scripts/run_retrieval_evaluation.sh configs/base.yaml 256
```

If the second argument is omitted, the script reads the selected size from
`outputs/analysis/chunk_size_pilot/selection.json`.

Confirmatory pairwise comparisons and the section-constrained-minus-global
structural analysis use only the configured test split:

```bash
# Run the current Granite confirmatory analysis without requiring Jina first.
sclc compare --model granite --chunk-size 128 --config configs/base.yaml
sclc analyse --model granite --chunk-size 128 --config configs/base.yaml

# After every Jina condition has also been evaluated, omit --model to analyse both.
sclc compare --chunk-size 128 --config configs/base.yaml
sclc analyse --chunk-size 128 --config configs/base.yaml
```

## Dense encoding behavior

- `fixed_dense` encodes continuous chunks independently.
- `section_isolated` encodes section-bounded target chunks independently.
- `section_constrained` encodes each parent section once and mean-pools the
  target token spans after contextualisation.
- `global` encodes each complete paper once and mean-pools the same target spans.
- No complete paper is truncated, split, or windowed for global late chunking.
- Vectors are L2-normalised and cached per paper.
- Query embeddings are shared across chunk sizes and conditions for each model.

If GPU memory is insufficient, reduce:

```yaml
dense:
  batch_size: 2
```

Contextual conditions process one paper or section at a time, so their memory
requirements are governed mainly by the contextual scope rather than the number
of target chunks pooled from it.

## Evidence policy

The primary relevance set is the union of distinct QASPER evidence paragraphs
across all acceptable annotations. Complete-support success occurs when all
paragraphs from at least one acceptable evidence set are retrieved.

Metrics prefixed with `best_` report the most favourable acceptable evidence set
as sensitivity analysis. Metrics prefixed with `union_complete_` apply the
stricter requirement that every paragraph in the evidence union is recovered.

Only non-whitespace character overlap counts toward relevance and evidence-span
coverage.

## Reproducibility and safeguards

- document IDs are always read as strings;
- random sampling and bootstrap seeds come from the YAML configuration;
- retrieval ties use deterministic retrieval-unit IDs;
- all chunk-dependent caches include chunk size and input fingerprints;
- query categories are required before retrieval;
- validation is used for chunk-size selection;
- test is reserved for confirmatory comparisons;
- invalid condition/model combinations and unsupported chunk sizes are rejected.

## Migration from v0.6.0

You do not need to rerun `prepare`, `profile`, or `sample` when upgrading.

Keep the completed file:

```text
data/retrieval_units/query_types.csv
```

Then construct the new namespaced chunk artefacts for 128, 256, and 512. Old
unscoped BM25 rankings and evaluations may be retained as an archive, but v0.7.0
does not treat them as pilot inputs because they lack chunk-size namespaces and
the expanded metrics.

## Development checks

```bash
pytest
ruff check .
mypy src
```

The packaged project was validated with `pytest` and Python bytecode compilation.
Ruff and mypy require the optional development dependencies and should be run in
the local development environment before the final experiment.

### Long-context GPU memory

The Granite R2 encoder is based on ModernBERT and supports long inputs, but the
manual/eager attention implementation can materialise a quadratic attention matrix.
The default configuration therefore fixes `dense.attn_implementation: sdpa`, which
uses PyTorch scaled-dot-product attention and substantially reduces peak memory.
The model also supports `flash_attention_2` when the optional Flash Attention package
is installed.

A CUDA OOM during `section_constrained` or `global` is not normally fixed by lowering
`dense.batch_size`: each section or complete document is already encoded one at a
time. If SDPA still exceeds the available VRAM, run the contextual condition on a
larger GPU. Do not truncate or window complete documents, because that changes the
global contextual-scope condition.

### FlexAttention on memory-constrained GPUs

Transformers can route ModernBERT attention through PyTorch FlexAttention by setting:

```yaml
dense:
  device: cuda
  dtype: float16
  attn_implementation: flex_attention
```

This is useful when SDPA falls back to a quadratic attention allocation for a long
ModernBERT scope. FlexAttention remains an exact attention implementation and does
not truncate or window the input. The first forward pass may spend additional time
compiling kernels. Use the same backend and dtype for all dense conditions in the
final controlled experiment.

Changing `dense.attn_implementation` changes the dense-cache fingerprint. Rebuild all
dense conditions and shared query embeddings with `--overwrite` so every condition
uses the same backend. BM25 outputs are unaffected.

## Audit the complete QASPER collection

After preparing and profiling the corpus, inspect all usable QASPER questions for
cross-section evidence requirements:

```bash
sclc qasper-audit --config configs/base.yaml
```

This produces a read-only collection audit under
`outputs/analysis/qasper_collection_audit/`. It does not modify the selected
sample or run retrieval. See `docs/qasper_collection_audit.md` for the strict
candidate definition and split safeguards.

## Freeze the strict cross-section challenge

After `sclc qasper-audit`, freeze the supplementary test-only challenge and
create a blinded review sheet:

```bash
sclc qasper-challenge --config configs/base.yaml --overwrite
```

This leaves the primary sample unchanged and writes deterministic challenge
manifests under `outputs/analysis/qasper_cross_section_challenge/`. Complete the
blinded review before running any new challenge retrieval. See
`docs/qasper_cross_section_challenge.md`.
