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

run_trace() {
  local name=$1
  local chunk=$2
  local conditions=$3
  local out="$BASE/$name"
  if [[ -f "$out/run.json" ]]; then
    return
  fi
  export MLLM_COMM_CONTROL="$ROOT/$BASE/${name}_control.json"
  export MLLM_COMM_OUTPUT="$ROOT/$out/raw_stage"
  "$PY" -m poc_mllm_ep_collective_granularity.scripts.run_serving \
    --manifest "$MANIFEST" \
    --output "$out" \
    --control "$MLLM_COMM_CONTROL" \
    --chunk-size "$chunk" \
    --conditions "$conditions" \
    --repeats 3 \
    --decode-tokens 64 \
    --injection-progress 16 \
    --trace
}

# The initial FULL/CP2048 traces predated the explicit caller-visible outer
# iteration tag. These append-only reruns preserve both internal and outer IDs.
run_trace full_trace_aligned 0 DECODE_ONLY,VISUAL_PREFILL,TOKEN_MATCHED_TEXT
run_trace cp2048_trace_aligned 2048 DECODE_ONLY,VISUAL_PREFILL

if [[ ! -f "$BASE/cp256_trace_aligned/run.json" ]]; then
  export MLLM_COMM_CONTROL="$ROOT/$BASE/cp256_trace_aligned_control.json"
  export MLLM_COMM_OUTPUT="$ROOT/$BASE/cp256_trace_aligned/raw_stage"
  "$PY" -m poc_mllm_ep_collective_granularity.scripts.run_serving \
    --manifest "$MANIFEST" \
    --output "$BASE/cp256_trace_aligned" \
    --control "$MLLM_COMM_CONTROL" \
    --chunk-size 256 \
    --conditions DECODE_ONLY,VISUAL_PREFILL \
    --repeats 3 \
    --decode-tokens 64 \
    --injection-progress 16 \
    --trace \
    --trace-through-decode
fi
