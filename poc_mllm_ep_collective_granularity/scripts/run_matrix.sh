#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/esjung/MLLM-EP-modality-discovery}"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=2,3,6,7
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export MLLM_COMM_MOTIVATION_ENABLE=1
export PYTHONPATH="$ROOT/poc_mllm_ep_collective_granularity/hooks:$ROOT"
export VLLM_NO_USAGE_STATS=1

PY=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
MANIFEST=poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json
BASE=poc_mllm_ep_collective_granularity/artifacts

run_case() {
  local name=$1
  local chunk=$2
  local repeats=$3
  local conditions=$4
  local trace=$5
  local out="$BASE/$name"
  if [[ -f "$out/run.json" ]]; then
    return
  fi
  export MLLM_COMM_CONTROL="$ROOT/$BASE/${name}_control.json"
  export MLLM_COMM_OUTPUT="$ROOT/$out/raw_stage"
  local trace_arg=()
  if [[ "$trace" == "true" ]]; then
    trace_arg=(--trace)
  fi
  "$PY" -m poc_mllm_ep_collective_granularity.scripts.run_serving \
    --manifest "$MANIFEST" \
    --output "$out" \
    --control "$MLLM_COMM_CONTROL" \
    --chunk-size "$chunk" \
    --conditions "$conditions" \
    --repeats "$repeats" \
    --decode-tokens 64 \
    --injection-progress 16 \
    "${trace_arg[@]}"
}

# Full baseline: clean confirmation is collected separately with ten repeats.
run_case full_trace 0 3 DECODE_ONLY,VISUAL_PREFILL,TOKEN_MATCHED_TEXT true

for chunk in 2048 1024 512 256; do
  run_case "cp${chunk}_clean" "$chunk" 3 DECODE_ONLY,VISUAL_PREFILL false
  run_case "cp${chunk}_trace" "$chunk" 3 DECODE_ONLY,VISUAL_PREFILL true
done
