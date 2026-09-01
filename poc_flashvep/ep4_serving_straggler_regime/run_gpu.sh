#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export FLASHVEP_SERVING_PROBE=1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
venv=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
workspace=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
model=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
result=${1:?usage: run_gpu.sh RESULT_DIR}
export PATH="$venv/bin:$PATH" NVSHMEM_DIR="$workspace/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo_root/poc_flashvep/ep4_serving_straggler_regime/hooks:$repo_root:$repo_root/poc_flashvep/live_traffic_matrix_validation/hooks:${PYTHONPATH:-}"
exec "$venv/bin/python" "$repo_root/poc_flashvep/ep4_serving_straggler_regime/run_serving.py" \
  --model-path "$model" --output-dir "$result" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --warmups "${WARMUPS:-1}" --iterations "${ITERATIONS:-2}"
