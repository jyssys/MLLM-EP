#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export PYTHONUNBUFFERED=1
export FLASHVEP_SERVING_PROBE=1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
venv=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
workspace=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
export PATH="$venv/bin:$PATH" NVSHMEM_DIR="$workspace/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo_root/poc_flashvep/ep4_serving_straggler_regime/hooks:$repo_root:$repo_root/poc_flashvep/live_traffic_matrix_validation/hooks:${PYTHONPATH:-}"
MODEL_PATH="${MODEL_PATH:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}"
exec "$venv/bin/python" "$repo_root/poc_flashvep/ep4_serving_straggler_regime/run_mixed.py" \
  --model-path "$MODEL_PATH" --output-dir "$1" \
  --prefill-count "${PREFILL_COUNT:-3}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
  --warmups "${WARMUPS:-1}" --iterations "${ITERATIONS:-2}"
