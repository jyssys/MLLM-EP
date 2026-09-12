#!/usr/bin/env bash
set -euo pipefail

root=/home/esjung/MLLM-EP-llada2-flash-100b
runner="$root/poc_llada2_flash_ep/scripts/run_clean_point.sh"

# Fixed ABBA-style order. These are independent engine restarts and validate
# the non-monotonic M=256/M=512 crossover before any method interpretation.
matrix=(
  "tp4 8 2" "ep4 8 2" "ep4 16 2" "tp4 16 2"
  "ep4 8 3" "tp4 8 3" "tp4 16 3" "ep4 16 3"
)
for point in "${matrix[@]}"; do
  read -r topology mini repeat <<<"$point"
  bash "$runner" "$topology" 16 "$repeat" "dynamic${mini}" 32
done
