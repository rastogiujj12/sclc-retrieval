#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/base.yaml}"
MODEL="${2:-}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi

if [[ "${MODEL}" != "granite" && "${MODEL}" != "jina" ]]; then
  echo "Usage: $0 [config-path] {granite|jina}" >&2
  exit 2
fi

if ! command -v sclc >/dev/null 2>&1; then
  echo "The 'sclc' command is not available. Install the project first:" >&2
  echo "  python -m pip install -e '.[dev]'" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration file not found: ${CONFIG_PATH}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

if [[ ! -f data/processed/documents.jsonl ]]; then
  echo "Prepared QASPER documents are missing." >&2
  echo "Run ./scripts/prepare_primary_data.sh first." >&2
  exit 2
fi

if [[ ! -f data/subsets/selected_documents.csv ]]; then
  echo "Frozen sample manifest is missing: data/subsets/selected_documents.csv" >&2
  exit 2
fi

if [[ ! -f data/retrieval_units/query_types.csv ]]; then
  echo "Frozen query-type labels are missing: data/retrieval_units/query_types.csv" >&2
  exit 2
fi

SIZES=(128 256 512)
DENSE_CONDITIONS=(fixed_dense section_isolated section_constrained global)

for SIZE in "${SIZES[@]}"; do
  echo
  echo "=== Retrieval-unit size: ${SIZE} tokens ==="

  sclc build-units --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"

  sclc encode --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  sclc retrieve --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  sclc evaluate --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"

  for CONDITION in "${DENSE_CONDITIONS[@]}"; do
    echo "--- ${MODEL}/${CONDITION}/${SIZE} ---"
    sclc encode \
      --condition "${CONDITION}" \
      --model "${MODEL}" \
      --retrieval-unit-size "${SIZE}" \
      --config "${CONFIG_PATH}"
    sclc retrieve \
      --condition "${CONDITION}" \
      --model "${MODEL}" \
      --retrieval-unit-size "${SIZE}" \
      --config "${CONFIG_PATH}"
    sclc evaluate \
      --condition "${CONDITION}" \
      --model "${MODEL}" \
      --retrieval-unit-size "${SIZE}" \
      --config "${CONFIG_PATH}"
  done

  sclc compare \
    --model "${MODEL}" \
    --retrieval-unit-size "${SIZE}" \
    --config "${CONFIG_PATH}" \
    --overwrite
done

sclc retrieval-unit-size \
  --model "${MODEL}" \
  --retrieval-unit-sizes 128,256,512 \
  --config "${CONFIG_PATH}" \
  --overwrite

echo
echo "Primary experiment complete for ${MODEL}."
