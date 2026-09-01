#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR=${NVSHMEM_DIR:-/home/esjung/.cache/flashvep-deepep-v020/nvshmem}
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
venv=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
model=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
result=${1:?usage: run_gpu.sh result_dir topology mode scale delay_ms chunked}
topology=${2:?}
mode=${3:?}
scale=${4:?}
delay=${5:-0}
chunked=${6:-true}
mkdir -p "$(dirname "$result")"
exec "$venv/bin/python" "$repo_root/poc_flashvep/asap_sync_phenomenon_reproduction/run_asap.py" \
  --model-path "$model" --output-dir "$result" --topology "$topology" --mode "$mode" \
  --scale "$scale" --delay-ms "$delay" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --chunked-prefill "$chunked" --warmups "${WARMUPS:-1}" --iterations "${ITERATIONS:-1}"
