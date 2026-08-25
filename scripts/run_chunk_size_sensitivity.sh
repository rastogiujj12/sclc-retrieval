#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/base.yaml}"
MODEL="${2:-granite}"

if [[ "$MODEL" != "granite" && "$MODEL" != "jina" ]]; then
  echo "Model must be granite or jina." >&2
  exit 2
fi

CHUNK_SIZES=(128 256 512)
CONDITIONS=(fixed_dense section_isolated section_constrained global)

echo "Running chunk-size sensitivity for model: $MODEL"
echo "Config: $CONFIG_PATH"

for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
  echo
  echo "=== Chunk size: ${CHUNK_SIZE} ==="

  sclc chunk \
    --chunk-size "$CHUNK_SIZE" \
    --config "$CONFIG_PATH"

  for CONDITION in "${CONDITIONS[@]}"; do
    echo
    echo "--- ${MODEL}/${CONDITION}/${CHUNK_SIZE} ---"
    sclc encode \
      --condition "$CONDITION" \
      --model "$MODEL" \
      --chunk-size "$CHUNK_SIZE" \
      --config "$CONFIG_PATH"
    sclc retrieve \
      --condition "$CONDITION" \
      --model "$MODEL" \
      --chunk-size "$CHUNK_SIZE" \
      --config "$CONFIG_PATH"
    sclc evaluate \
      --condition "$CONDITION" \
      --model "$MODEL" \
      --chunk-size "$CHUNK_SIZE" \
      --config "$CONFIG_PATH"
  done
done

sclc chunk-size-sensitivity \
  --model "$MODEL" \
  --chunk-sizes 128,256,512 \
  --config "$CONFIG_PATH" \
  --overwrite

echo
echo "Chunk-size sensitivity complete for $MODEL."
