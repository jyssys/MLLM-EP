#!/usr/bin/env bash
set -euo pipefail

repo="/home/esjung/MLLM-EP-long-reasoning-signal"
poc="$repo/poc_mllm_vision_conditioned_decode_reduction"
python_bin="/home/esjung/.venvs/flashvep-deepep-v020/bin/python"
manifest="$poc/artifacts/frozen_vector_capture_cohort.json"
run_name="${1:-vector_capture_discovery_v1}"
output="$poc/artifacts/$run_name"
control="$poc/artifacts/${run_name}_control.json"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$repo:/home/esjung/MLLM-EP-github${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0

cleanup() {
  PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_reduction.gpu_guard \
    start --record "$poc/artifacts/gpu_burner_restart.json" || true
}
trap cleanup EXIT INT TERM

PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_reduction.gpu_guard \
  stop --record "$poc/artifacts/gpu_burner_stop.json"

cd "$repo"
"$python_bin" -m poc_mllm_vision_conditioned_decode_reduction.scripts.capture_decode_vectors_ep4 \
  --manifest "$manifest" \
  --output "$output" \
  --control "$control" \
  --limit 4 \
  --layers 4,12,24,36,44,47 \
  --max-model-len 4096 \
  --kv-cache-gib 2 \
  --wave-timeout-sec 3600
