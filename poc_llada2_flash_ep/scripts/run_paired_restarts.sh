#!/usr/bin/env bash
set -euo pipefail

root=/home/esjung/MLLM-EP-llada2-flash-100b
runner="$root/poc_llada2_flash_ep/scripts/run_clean_point.sh"

# Fixed, recorded randomization generated once with seed 744. Each line is an
# independent engine restart; paired analysis is performed within batch/repeat.
matrix=(
  "ep4 8 2" "tp4 8 2" "tp4 1 2" "ep4 1 2"
  "ep4 32 2" "tp4 32 2" "ep4 4 2" "tp4 4 2"
  "tp4 16 2" "ep4 16 2" "tp4 2 2" "ep4 2 2"
  "tp4 4 3" "ep4 4 3" "ep4 16 3" "tp4 16 3"
  "ep4 2 3" "tp4 2 3" "tp4 32 3" "ep4 32 3"
  "ep4 1 3" "tp4 1 3" "tp4 8 3" "ep4 8 3"
)

for point in "${matrix[@]}"; do
  read -r topology batch repeat <<<"$point"
  bash "$runner" "$topology" "$batch" "$repeat" dynamic4 32
done
