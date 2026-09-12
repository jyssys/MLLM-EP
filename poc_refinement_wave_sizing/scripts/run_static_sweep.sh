#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submitted_batch="${1:-32}"
generation="${2:-32}"

orders=(
  "8 1 32 4 16 2"
  "2 16 4 32 1 8"
  "32 4 8 2 16 1"
)

for repeat in 1 2 3; do
  read -r -a minis <<<"${orders[$((repeat - 1))]}"
  for mini in "${minis[@]}"; do
    "$script_dir/run_clean_point.sh" ep4 "$submitted_batch" "$mini" "$repeat" "$generation"
  done
done
