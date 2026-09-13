#!/usr/bin/env bash
set -euo pipefail

task="$1"
sample_count="$2"
run_dir="$3"
repo=/home/esjung/MLLM-EP-dynamic-block-deep
if [[ "$sample_count" == "32" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval_32"
elif [[ "$sample_count" == "8" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval"
elif [[ "$sample_count" == "1" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/smoke_eval"
else
  exit 2
fi
prediction=$(find "$run_dir" -maxdepth 1 -type f -name '*.jsonl' -print -quit)
test -n "$prediction"
PYTHONDONTWRITEBYTECODE=1 python \
  "$repo/poc_llada2_flash_ep/scripts/evaluate_bounded_quality.py" \
  --task "$task" --predictions "$prediction" \
  --truth "$data_dir/${task}_${sample_count}_truth.json" \
  --output "$run_dir/quality.json"
