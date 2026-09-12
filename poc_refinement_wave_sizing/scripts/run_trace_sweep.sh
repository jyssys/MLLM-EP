#!/usr/bin/env bash
set -euo pipefail

root=/home/esjung/MLLM-EP-raws
for mini in 16 32 8 4 2 1; do
  bash "$root/poc_refinement_wave_sizing/scripts/run_shape_trace.sh" ep4 32 "$mini" shape 32
done
