# Section-Constrained Late Chunking for Academic Retrieval

This repository contains the experimental software artefact for the MSc dissertation:

> **Section-Constrained Late Chunking for Academic Retrieval: An Ablation Study**

The study treats **contextual encoding scope** as an independent retrieval-design
variable. It asks whether a target retrieval unit in a structured academic paper
should be encoded using only its own text, its parent top-level section, or the
complete prepared paper.

## Study design

Five retrieval conditions are evaluated at canonical target retrieval-unit sizes
of **128, 256, and 512 tokens**, with zero overlap.

| Condition | Retrieval-unit construction | Context visible during encoding |
|---|---|---|
| `bm25` | Continuous fixed-size units | Not applicable (lexical baseline) |
| `fixed_dense` | Continuous fixed-size units | Retrieval unit only |
| `section_isolated` | Section-bounded units | Retrieval unit only |
| `section_constrained` | Section-bounded units | Parent top-level section |
| `global` | Section-bounded units | Complete prepared paper |

The three section-bounded contextual conditions use the **same target text
spans**. Their controlled difference is the amount of surrounding document text
visible while the target representation is formed.

All dense conditions use explicit target-token mean pooling, L2-normalised
vectors, and exact dot-product ranking. The study does not use approximate
nearest-neighbour search, reranking, query expansion, answer generation, or
model fine-tuning.

## Dataset and frozen sample

The experiments use QASPER. The source collection contains **1,585 papers and
5,049 questions**. After preparation and evidence alignment, **4,295 questions**
have usable textual evidence.

The frozen experimental sample contains **200 papers and 554 questions**:

- **Cross-model core:** 150 papers / 425 questions, evaluated with Granite and Jina.
- **Granite extension:** 50 longer papers / 129 questions, evaluated with Granite only.

The primary reported results use the original QASPER test portion contained in
the frozen sample:

- **Granite:** 55 papers / 180 questions.
- **Jina:** 42 cross-model papers / 137 questions.
- **Granite extension:** 13 papers / 43 questions.

Retrieval is performed **within the question's associated paper**. The task is
therefore retrieval-unit ranking within a known relevant document rather than
collection-wide document discovery.

The final query-type coding across the frozen 554-question sample is:

- factual: 432;
- section-specific: 66;
- multi-hop: 21;
- synthesis: 35.

## Embedding models

- `ibm-granite/granite-embedding-311m-multilingual-r2`
  - configured maximum context: 32,768 tokens
- `jinaai/jina-embeddings-v3-hf`
  - configured maximum context: 8,192 tokens
  - passage adapter: `retrieval_passage`
  - query adapter: `retrieval_query`

Granite tokenisation defines the canonical 128/256/512-token retrieval-unit
boundaries. Jina reuses the resulting character spans so both embedding models
rank the same underlying text.

## Main findings reproduced by the artefact

The dissertation's main result is asymmetric:

- fixed-size dense retrieval is the strongest overall condition;
- Section-Constrained Late Chunking does **not** consistently outperform
  section-isolated encoding;
- Section-Constrained Late Chunking achieves significantly higher nDCG@5 than
  global late chunking at every retrieval-unit size for both embedding models
  in the primary analysis.

The software also supports the reported longer-paper, query-type, retrieval-unit-size,
and exploratory cross-section analyses.

## Installation

Python **3.11 or 3.12** is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Dense encoding is designed for CUDA GPUs. CPU execution is possible for smaller
stages but is not practical for the long-context dense experiment.

Run the regression suite with:

```bash
pytest
```

The repository currently contains **46 regression tests**.

## Reproducing the primary experiment

The repository commits the **frozen document sample** and the final manual
query-type labels. A reproduction therefore does not need to resample the
collection before running the reported experiment.

### 1. Prepare the QASPER source documents

From a fresh clone, install the project and run:

```bash
./scripts/prepare_primary_data.sh configs/base.yaml
```

This reconstructs the prepared QASPER papers and removes reference sections. It
leaves the committed dissertation sample untouched. The frozen sample manifest
is:

```text
data/subsets/selected_documents.csv
```

To independently verify the deterministic sampling procedure, run:

```bash
sclc profile --config configs/base.yaml
sclc sample --config configs/base.yaml
git diff -- data/subsets/selected_documents.csv
```

A clean diff confirms that the regenerated selection matches the committed
frozen sample.

### 2. Query-type coding

The final manually coded query labels are committed at:

```text
data/retrieval_units/query_types.csv
```

The corresponding detailed coding record is retained at
`data/retrieval_units/query_type_coding_record.csv`, and the coding guide is in
`docs/query_type_coding_guide.md`. The dissertation used
single-researcher coding with predefined criteria; no independent second coder
or inter-rater reliability estimate was used.

### 3. Run the primary retrieval experiment

Run one embedding model at a time so long-context stages can resume safely from
validated document-level outputs:

```bash
./scripts/run_primary_experiment.sh configs/base.yaml granite
./scripts/run_primary_experiment.sh configs/base.yaml jina
```

The runner checks the required frozen inputs before starting. It evaluates all
five conditions at 128, 256, and 512 tokens, runs the paired condition
comparisons at each size, and performs the retrieval-unit-size interaction
analysis.

The equivalent individual commands are available through:

```bash
sclc --help
```

For example, retrieval units for one size can be constructed with:

```bash
sclc build-units \
  --retrieval-unit-size 256 \
  --config configs/base.yaml
```

### 4. Retrieval-unit-size analysis

After the required dense evaluations exist, the analysis can be rerun independently:

```bash
sclc retrieval-unit-size \
  --model granite \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

Repeat with `--model jina` for the cross-model sample. See
`docs/retrieval_unit_size_analysis.md` for the output definitions.

## Pairwise statistical comparisons

The primary comparison family contains four condition pairs. Signed differences
follow **first condition minus second condition**; positive values favour the
first named condition and negative values favour the second.

The reported dense directions are:

- section-isolated minus fixed-size dense;
- section-constrained minus section-isolated;
- global minus section-constrained.

Statistical uncertainty is estimated with **10,000 paired document-level
bootstrap resamples**. Holm correction is applied within the predefined family
for each model, analysis set, question group, retrieval-unit size, and metric.

## Evaluation metrics

The six principal dissertation metrics are:

- nDCG@5;
- Recall@5;
- evidence-paragraph recall@5 (ER@5);
- complete evidence recovery@5 (CE@5);
- evidence-paragraph recall within 1,024 retrieved tokens;
- complete evidence recovery within 2,048 retrieved tokens.

QASPER may provide multiple acceptable evidence sets for one question. Complete
evidence recovery succeeds when every paragraph in **any one complete acceptable
set** has been recovered.

The evaluator also retains additional diagnostic metrics used to validate the
pipeline. Complete rankings are stored so fixed-rank and token-budget measures
can be calculated without repeating retrieval.

## Cross-section challenge

A separate exploratory audit identifies questions whose complete acceptable
evidence necessarily spans more than one top-level section. The dissertation
release commits the completed manual review and the accepted challenge document
manifest, so the reported challenge can be rerun directly after QASPER
preparation.

To regenerate the original audit/review inputs from the complete prepared
collection, use:

```bash
sclc qasper-audit --config configs/base.yaml
sclc qasper-challenge --config configs/base.yaml --overwrite
```

The complete audit identified **302 strict candidates across 253 papers**. The
original QASPER test split contained 53 strict candidates, which were manually
reviewed. The final review accepted **23 questions across 20 papers**:

- 18 questions from 15 cross-model papers;
- 5 questions from 5 Granite-extension papers.

The completed review decisions are committed at:

```text
data/subsets_cross_section_challenge/review_decisions.csv
```

The dissertation reports the 18 accepted cross-model questions with Jina at all
three retrieval-unit sizes. Reproduce that reported challenge with:

```bash
./scripts/run_cross_section_challenge.sh \
  configs/cross_section_challenge.yaml \
  jina
```

The complete audit, review, and challenge workflow is documented in
`docs/cross_section_challenge.md`.

## Reproducibility controls

The pipeline records and validates configuration fingerprints and source
manifests before reusing generated outputs. Important controls include:

- project seed 42;
- deterministic document-level sampling;
- canonical target sizes of 128/256/512 tokens with zero overlap;
- identical section-bounded target spans across the three contextual conditions;
- no silent truncation or sliding-window approximation for global late chunking;
- shared query representations within each embedding model;
- explicit target-token mean pooling for all dense conditions;
- exact ranking rather than approximate nearest-neighbour search;
- deterministic ranking tie-breaking;
- document-level paired bootstrap resampling;
- restart-safe document-level dense-encoding outputs.

Generated encodings, rankings, evaluation files, bootstrap arrays, and model
caches are intentionally excluded from Git. They can be reconstructed from the
frozen inputs and configuration.

### Software, GPU, and model snapshot

For an archival release, capture the local experiment environment with:

```bash
python scripts/capture_environment.py \
  --config configs/base.yaml \
  --output reproducibility/environment.json
```

The script records Python/package versions, CUDA and GPU information, the Git
commit, configured model identifiers, cached Hugging Face commit references when
they are recoverable from the local cache, the configured QASPER source URLs,
and SHA-256 hashes of the frozen dissertation inputs. See
`reproducibility/README.md`.

The exact model commit identifiers used by the completed dissertation runs were
not stored in the archived source tree, so this repository does not invent them
after the fact. If the original Hugging Face cache is available, the capture
script can recover the cached `main` revisions for the archival release.

## Repository structure

```text
configs/
  base.yaml                         Primary experiment configuration
  cross_section_challenge.yaml      Cross-section challenge configuration

data/
  subsets/                          Frozen 200-paper sample
  subsets_cross_section_challenge/  Manual review and challenge sample
  retrieval_units/                  Final query-type coding artefacts

docs/                               Method and audit documentation
examples/                           Small input templates
reproducibility/                    Environment-capture guidance and snapshot
scripts/                            End-to-end and provenance utilities
src/sclc/                           Experiment implementation
tests/                              Regression tests
```

Generated outputs remain separated by retrieval-unit size on disk so runs at
different sizes cannot overwrite one another:

```text
outputs/
  encodings/chunk_128/ ... chunk_512/
  rankings/chunk_128/  ... chunk_512/
  evaluation/chunk_128/ ... chunk_512/
  analysis/retrieval_unit_size/<model>/
```

The `chunk_<size>` directory name is an internal storage namespace; in the
dissertation and public documentation these values are referred to as
**retrieval-unit sizes**.

## Development checks

```bash
pytest
ruff check .
mypy src
python scripts/check_release.py
```

`check_release.py` verifies the frozen sample/query-type/challenge counts, key
experiment settings, release-version consistency, and absence of the removed
pilot/version-history terminology.

## Third-party data and models

The MIT licence in this repository applies to the original source code in this
project. QASPER and the external embedding models are separate third-party
resources and remain subject to their own licences and terms. Model weights and
QASPER source data are not redistributed by this repository.

## Citation

Citation metadata for the software artefact is provided in `CITATION.cff`.

**Ujjwal Rastogi**  
MSc in Computing in Big Data Analytics and Artificial Intelligence  
Atlantic Technological University, 2026
