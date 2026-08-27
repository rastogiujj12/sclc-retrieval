# Section-Constrained Late Chunking for Academic Retrieval

This repository contains the experimental artefact for the MSc dissertation
**Section-Constrained Late Chunking for Academic Retrieval: An Ablation Study**.

The study asks whether contextual encoding for retrieval representations in
structured academic papers should stop at top-level section boundaries rather
than extend across the complete prepared paper.

## Experimental design

Five retrieval conditions are evaluated at **128, 256, and 512 tokens**, with
zero overlap.

| Condition | Retrieval-unit construction | Context visible during encoding |
|---|---|---|
| `bm25` | Continuous fixed-size units | Not applicable |
| `fixed_dense` | Continuous fixed-size units | Retrieval unit only |
| `section_isolated` | Section-bounded units | Retrieval unit only |
| `section_constrained` | Section-bounded units | Parent top-level section |
| `global` | Section-bounded units | Complete prepared paper |

The three section-bounded contextual conditions use identical target spans;
only contextual encoding scope changes.

## Dataset and frozen sample

The experiments use QASPER. The frozen experimental sample contains **200
papers and 554 questions**:

- 150 cross-model papers / 425 questions, evaluated with Granite and Jina;
- 50 Granite-extension papers / 129 questions, evaluated only with Granite.

Primary results use the original QASPER test portion within this sample:
55 papers / 180 questions for Granite and 42 papers / 137 questions for Jina.
Retrieval is performed within each question's associated paper.

## Models

- `ibm-granite/granite-embedding-311m-multilingual-r2`
- `jinaai/jina-embeddings-v3-hf`

Granite tokenisation defines the canonical retrieval-unit boundaries. Jina
reuses the same character spans.

## Installation

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests with:

```bash
pytest
```

## Reproducing the primary experiment

Prepare QASPER while retaining the committed frozen sample:

```bash
./scripts/prepare_primary_data.sh configs/base.yaml
```

Run each embedding model:

```bash
./scripts/run_primary_experiment.sh configs/base.yaml granite
./scripts/run_primary_experiment.sh configs/base.yaml jina
```

The final query-type labels are committed at
`data/retrieval_units/query_types.csv`.

## Retrieval-unit-size analysis

```bash
sclc retrieval-unit-size \
  --model granite \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

Signed differences use **first condition minus second condition**. Positive
values favour the first named condition; negative values favour the second.

## Evaluation metrics

The six principal dissertation metrics are:

- nDCG@5
- Recall@5
- evidence-paragraph recall@5
- complete evidence recovery@5
- evidence-paragraph recall within 1,024 retrieved tokens
- complete evidence recovery within 2,048 retrieved tokens

Pairwise uncertainty is estimated with 10,000 paired document-level bootstrap
resamples and Holm correction.

## Cross-section challenge

The full audit identified 302 strict cross-section candidates across 253
papers. Fifty-three QASPER test candidates were manually reviewed; 23 were
accepted. The dissertation reports the 18 accepted cross-model questions from
15 papers using Jina at all three retrieval-unit sizes.

```bash
./scripts/run_cross_section_challenge.sh \
  configs/cross_section_challenge.yaml \
  jina
```

See `docs/cross_section_challenge.md` for details.

## Reproducibility

Generated model caches, encodings, rankings, evaluations, and bootstrap arrays
are intentionally excluded from Git. Frozen sample and coding inputs are kept
in the repository.

Before a release, run:

```bash
pytest
ruff check .
mypy src
python scripts/check_release.py
```

Use `scripts/capture_environment.py` on the experiment machine to record the
Python/package/CUDA/GPU environment and recover cached Hugging Face revisions
when available.

## Repository structure

```text
configs/          Experiment configurations
data/             Frozen sample and coding artefacts
docs/             Method and analysis documentation
reproducibility/  Release/environment records
scripts/          Reproduction and release helpers
src/sclc/         Experiment implementation
tests/            Regression tests
```

## Dissertation

Ujjwal Rastogi  
MSc in Computing in Big Data Analytics and Artificial Intelligence  
Atlantic Technological University, 2026
