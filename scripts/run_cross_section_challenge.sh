#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/cross_section_challenge.yaml}"
MODEL="${2:-jina}"
ACCEPTED_QUERIES="${3:-data/subsets_cross_section_challenge/review_decisions.csv}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi
if [[ "${ACCEPTED_QUERIES}" != /* ]]; then
  ACCEPTED_QUERIES="${REPO_ROOT}/${ACCEPTED_QUERIES}"
fi

if [[ "${MODEL}" != "granite" && "${MODEL}" != "jina" ]]; then
  echo "Usage: $0 [config-path] {granite|jina} [review-decisions.csv]" >&2
  exit 2
fi

if ! command -v sclc >/dev/null 2>&1; then
  echo "The 'sclc' command is not available. Install the project first:" >&2
  echo "  python -m pip install -e '.[dev]'" >&2
  exit 2
fi

for REQUIRED in "${CONFIG_PATH}" "${ACCEPTED_QUERIES}"; do
  if [[ ! -f "${REQUIRED}" ]]; then
    echo "Required file not found: ${REQUIRED}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"

if [[ ! -f data/processed/documents.jsonl ]]; then
  echo "Prepared QASPER documents are missing." >&2
  echo "Run ./scripts/prepare_primary_data.sh first." >&2
  exit 2
fi

if [[ ! -f data/subsets_cross_section_challenge/selected_documents.csv ]]; then
  echo "Frozen challenge document manifest is missing." >&2
  exit 2
fi

SIZES=(128 256 512)
CONDITIONS=(section_isolated section_constrained global)

for SIZE in "${SIZES[@]}"; do
  echo
  echo "=== Cross-section challenge: ${MODEL}, ${SIZE} tokens ==="
  sclc build-units --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"

  for CONDITION in "${CONDITIONS[@]}"; do
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
done

sclc challenge-analyse \
  --model "${MODEL}" \
  --accepted-queries "${ACCEPTED_QUERIES}" \
  --retrieval-unit-sizes 128,256,512 \
  --config "${CONFIG_PATH}" \
  --overwrite

echo
echo "Cross-section challenge complete for ${MODEL}."
