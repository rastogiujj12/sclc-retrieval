#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/base.yaml}"
CHUNK_SIZE="${2:-}"

if [[ -z "$CHUNK_SIZE" ]]; then
  SELECTION_PATH="outputs/analysis/chunk_size_pilot/selection.json"
  if [[ ! -f "$SELECTION_PATH" ]]; then
    echo "Pass the selected chunk size as argument 2, or run the pilot first." >&2
    exit 1
  fi
  CHUNK_SIZE="$(python - "$SELECTION_PATH" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["selected_chunk_size_tokens"])
PY
)"
fi

echo "Using selected chunk size: ${CHUNK_SIZE} tokens"

sclc retrieve \
  --condition bm25 \
  --chunk-size "$CHUNK_SIZE" \
  --config "$CONFIG_PATH"
sclc evaluate \
  --condition bm25 \
  --chunk-size "$CHUNK_SIZE" \
  --config "$CONFIG_PATH"

for model in granite jina; do
  for condition in fixed_dense section_isolated section_constrained global; do
    sclc retrieve \
      --condition "$condition" \
      --model "$model" \
      --chunk-size "$CHUNK_SIZE" \
      --config "$CONFIG_PATH"
    sclc evaluate \
      --condition "$condition" \
      --model "$model" \
      --chunk-size "$CHUNK_SIZE" \
      --config "$CONFIG_PATH"
  done
done

sclc compare --chunk-size "$CHUNK_SIZE" --config "$CONFIG_PATH"
sclc analyse --chunk-size "$CHUNK_SIZE" --config "$CONFIG_PATH"
