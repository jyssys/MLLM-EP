#!/usr/bin/env bash
set -euo pipefail

root=/home/esjung/MLLM-EP-raws
runner="$root/poc_refinement_wave_sizing/scripts/run_clean_point.sh"
bash "$runner" ep4 32 16 4 32
bash "$runner" ep4 32 32 4 32
bash "$runner" ep4 32 32 5 32
bash "$runner" ep4 32 16 5 32
