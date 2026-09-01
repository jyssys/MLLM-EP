#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR=${NVSHMEM_DIR:-/home/esjung/.cache/flashvep-deepep-v020/nvshmem}
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$repo_root/poc_flashvep/asap_sync_phenomenon_reproduction/hooks:$repo_root:$repo_root/poc_flashvep/ep4_serving_straggler_regime/hooks:${PYTHONPATH:-}"
venv=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
model=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
result=${1:?usage: run_gpu.sh result_dir topology mode scale delay_ms chunked}
topology=${2:?}
mode=${3:?}
scale=${4:?}
delay=${5:-0}
chunked=${6:-true}
mkdir -p "$(dirname "$result")"
chunk_flag=--chunked-prefill
if [[ "$chunked" != "true" ]]; then chunk_flag=--no-chunked-prefill; fi
extra_args=()
if [[ -n "${DELAY_SWEEP:-}" ]]; then
  # Space-separated values are kept in the preregistered order supplied by
  # the caller, e.g. DELAY_SWEEP='0 0.5 1 2'.
  read -r -a sweep_values <<< "$DELAY_SWEEP"
  extra_args+=(--delay-sweep "${sweep_values[@]}")
fi
exec "$venv/bin/python" "$repo_root/poc_flashvep/asap_sync_phenomenon_reproduction/run_asap.py" \
  --model-path "$model" --output-dir "$result" --topology "$topology" --mode "$mode" \
  --scale "$scale" --delay-ms "$delay" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  "$chunk_flag" --warmups "${WARMUPS:-1}" --iterations "${ITERATIONS:-1}" "${extra_args[@]}"
