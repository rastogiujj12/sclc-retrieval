#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/base.yaml}"
CHUNK_SIZES=(128 256 512)

if [[ ! -f data/retrieval_units/query_types.csv ]]; then
  echo "Missing data/retrieval_units/query_types.csv." >&2
  echo "Complete the blind query-type coding before retrieval." >&2
  exit 1
fi

for size in "${CHUNK_SIZES[@]}"; do
  echo "== Building ${size}-token retrieval units =="
  sclc chunk --chunk-size "$size" --config "$CONFIG_PATH"

  echo "== BM25: ${size} tokens =="
  sclc encode --condition bm25 --chunk-size "$size" --config "$CONFIG_PATH"
  sclc retrieve --condition bm25 --chunk-size "$size" --config "$CONFIG_PATH"
  sclc evaluate --condition bm25 --chunk-size "$size" --config "$CONFIG_PATH"

done

for size in "${CHUNK_SIZES[@]}"; do
  echo "== Granite fixed dense: ${size} tokens =="
  sclc encode \
    --condition fixed_dense \
    --model granite \
    --chunk-size "$size" \
    --config "$CONFIG_PATH"
  sclc retrieve \
    --condition fixed_dense \
    --model granite \
    --chunk-size "$size" \
    --config "$CONFIG_PATH"
  sclc evaluate \
    --condition fixed_dense \
    --model granite \
    --chunk-size "$size" \
    --config "$CONFIG_PATH"
done

sclc select-chunk-size --config "$CONFIG_PATH"
