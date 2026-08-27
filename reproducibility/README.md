# Reproducibility snapshot

The experiment pipeline already fingerprints its prepared inputs, retrieval-unit
files, encodings, rankings, evaluations, and statistical-analysis configuration.
For an archival software release, the remaining useful provenance is the local
software/GPU environment and the exact Hugging Face model snapshots present in
the cache used for the run.

Capture that information from the experiment environment with:

```bash
python scripts/capture_environment.py \
  --config configs/base.yaml \
  --output reproducibility/environment.json
```

The script records:

- Python and operating-system information;
- versions of the principal Python dependencies;
- PyTorch CUDA/cuDNN information and detected GPU models;
- the current Git commit and working-tree status;
- SHA-256 hashes of the frozen sample, query-type labels, and challenge inputs;
- configured QASPER source URLs;
- configured model identifiers and revisions;
- the cached `main` commit for each Hugging Face model/tokenizer when it can be
  recovered from the configured local cache.

The script is deliberately offline: it does not resolve a newer remote `main`
revision. This avoids silently replacing the model snapshot that was actually
present in the experiment environment.

The release also includes `reproducibility/frozen_inputs.json`, which records
SHA-256 hashes and expected counts for the small frozen inputs committed with
the dissertation artefact. `python scripts/check_release.py` verifies those
files before release.

## Model revision note

The source configuration exposes explicit `revision` fields, but the archived
source tree does not contain a record of the exact Hugging Face commit hashes
used by the completed dissertation runs. Those identifiers should therefore not
be invented after the fact. If the original Hugging Face cache is still
available, run the capture script there and commit the resulting
`reproducibility/environment.json` with the archival release.

For a fresh reproduction where the historical cache is unavailable, pin the
model and canonical-tokenizer `revision` fields in the YAML configuration before
running the dense experiment, and archive the resulting environment snapshot.
