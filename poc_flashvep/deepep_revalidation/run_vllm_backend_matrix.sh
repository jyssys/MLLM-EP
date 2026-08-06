#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
RESULT_DIR=${1:?usage: run_vllm_backend_matrix.sh RESULT_DIR}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
WARMUPS=${WARMUPS:-5}
ITERATIONS=${ITERATIONS:-20}

export NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/deepep_revalidation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_DIR}/backend_proof"

run_case() {
  local name=$1
  local backend=$2
  local dbo=$3
  local dbo_args=()
  if [[ "${dbo}" == "on" ]]; then
    dbo_args+=(--enable-dbo)
  fi
  export FLASHVEP_DEEPEP_PROOF_DIR="${RESULT_DIR}/backend_proof/${name}"
  mkdir -p "${FLASHVEP_DEEPEP_PROOF_DIR}"
  "${DEEPEP_VENV}/bin/python" \
    "${REPO_ROOT}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
    --model-path "${MODEL}" \
    --output "${RESULT_DIR}/${name}.json" \
    --all2all-backend "${backend}" \
    "${dbo_args[@]}" \
    --warmups "${WARMUPS}" \
    --iterations "${ITERATIONS}" \
    2>&1 | tee "${RESULT_DIR}/${name}.log"
}

run_case stock_agrs_dbo_off allgather_reducescatter off
run_case stock_deepep_dbo_off deepep_high_throughput off
run_case stock_deepep_dbo_on deepep_high_throughput on
