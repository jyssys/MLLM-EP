#!/usr/bin/env bash
set -euo pipefail

repo="/home/esjung/MLLM-EP-long-reasoning-signal"
poc="$repo/poc_mllm_vision_conditioned_decode_damage_skipping"
python_bin="/home/esjung/.venvs/flashvep-deepep-v020/bin/python"
run_name="${1:-functional_damage_v1}"
limit_cases="${2:-}"
start_case="${3:-0}"
plan_name="${4:-frozen_functional_probe_plan.json}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$repo:/home/esjung/MLLM-EP-github${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0

cleanup() {
  PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_damage_skipping.gpu_guard \
    start --record "$poc/artifacts/gpu_burner_functional_restart.json" || true
}
trap cleanup EXIT INT TERM

PYTHONPATH="$repo" python -m poc_mllm_vision_conditioned_decode_damage_skipping.gpu_guard \
  stop --record "$poc/artifacts/gpu_burner_functional_stop.json"

cd "$repo"
cmd=("$python_bin" -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.run_functional_probe_ep4 \
  --plan "$poc/artifacts/manifests/$plan_name" \
  --output "$poc/artifacts/runs/$run_name" \
  --control "$poc/artifacts/${run_name}_control.json" \
  --max-model-len 32768 \
  --kv-cache-gib 2 \
  --wave-timeout-sec 1800 \
  --start-case "$start_case")
if [[ -n "$limit_cases" ]]; then
  cmd+=(--limit-cases "$limit_cases")
fi
"${cmd[@]}"
