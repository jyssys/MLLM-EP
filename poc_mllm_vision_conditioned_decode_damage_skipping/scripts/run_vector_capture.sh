#!/usr/bin/env bash
set -euo pipefail

repo="/home/esjung/MLLM-EP-long-reasoning-signal"
poc="$repo/poc_mllm_vision_conditioned_decode_damage_skipping"
python_bin="/home/esjung/.venvs/flashvep-deepep-v020/bin/python"
run_name="${1:-path_vector_capture_v1}"
start_index="${2:-0}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$repo:/home/esjung/MLLM-EP-github${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0

cleanup() {
  PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_damage_skipping.gpu_guard \
    start --record "$poc/artifacts/gpu_burner_restart.json" || true
}
trap cleanup EXIT INT TERM

PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_damage_skipping.gpu_guard \
  stop --record "$poc/artifacts/gpu_burner_stop.json"

cd "$repo"
"$python_bin" -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.capture_path_vectors_ep4 \
  --manifest "$poc/artifacts/manifests/frozen_damage_cohort.json" \
  --output "$poc/artifacts/runs/$run_name" \
  --control "$poc/artifacts/${run_name}_control.json" \
  --layers 4,12,24,36,44,47 \
  --steps 8,16,32,48,64,96,112 \
  --max-model-len 32768 \
  --kv-cache-gib 2 \
  --wave-timeout-sec 3600 \
  --start-index "$start_index"
