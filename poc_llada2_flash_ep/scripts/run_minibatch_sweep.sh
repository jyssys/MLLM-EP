#!/usr/bin/env bash
set -euo pipefail

root=/home/esjung/MLLM-EP-llada2-flash-100b
runner="$root/poc_llada2_flash_ep/scripts/run_clean_point.sh"

# Submitted batch 16 keeps enough independent requests available to expose
# model-forward microbatch geometry. Order alternates topology to avoid drift.
for mini in 1 2 4 8 16; do
  if (( mini % 4 == 0 )); then
    order=(tp4 ep4)
  else
    order=(ep4 tp4)
  fi
  for topology in "${order[@]}"; do
    bash "$runner" "$topology" 16 1 "dynamic${mini}" 32
  done
done
