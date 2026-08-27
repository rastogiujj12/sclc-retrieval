#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/base.yaml}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
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

echo "=== Preparing QASPER source documents ==="
sclc prepare --config "${CONFIG_PATH}"

echo
echo "Prepared documents are ready."
echo "The committed frozen sample manifest has not been modified."
echo "To independently reproduce the sampling step, run:"
echo "  sclc profile --config ${CONFIG_PATH}"
echo "  sclc sample --config ${CONFIG_PATH}"
