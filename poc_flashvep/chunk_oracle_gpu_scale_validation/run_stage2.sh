#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
WORKSPACE=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
RESULT=${1:?usage: run_stage2.sh RESULT_DIR}

export PATH="${VENV}/bin:${PATH}"
export NVSHMEM_DIR="${WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/vision_tile_motivation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
export FLASHVEP_VISION_TILE_CAPTURE_FIX=1
mkdir -p "$(dirname "${RESULT}")"
# Keep the tee log outside the output directory: long_capture.py creates the
# directory atomically with exist_ok=False, so logging must not pre-create it.

"${VENV}/bin/python" "${REPO_ROOT}/poc_flashvep/chunk_oracle_gpu_scale_validation/long_capture.py" \
  --model-path "${MODEL}" --output-dir "${RESULT}" 2>&1 | tee "${RESULT}.capture.log"
