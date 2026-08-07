#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULT_DIR=${1:?usage: run_dbo_final_numerical_gate.sh RESULT_DIR}
VENV=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
WORKSPACE=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
export NVSHMEM_DIR="${WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/deepep_revalidation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
unset FLASHVEP_DBO_CORRECTNESS_FIX FLASHVEP_DBO_CORRECTNESS_TRACE_DIR || true
mkdir -p "${RESULT_DIR}"

for mode in ${MODES:-off on}; do
  export FLASHVEP_DBO_LOCALIZATION_DIR="${RESULT_DIR}/tensors"
  export FLASHVEP_DBO_LOCALIZATION_MODE="${mode}"
  dbo_args=()
  [[ "${mode}" == "on" ]] && dbo_args+=(--enable-dbo)
  "${VENV}/bin/python" \
    "${REPO_ROOT}/poc_flashvep/deepep_revalidation/dbo_final_numerical_probe.py" \
    --model-path "${MODEL}" \
    --output "${RESULT_DIR}/distinct_red_dbo_${mode}.json" \
    "${dbo_args[@]}" 2>&1 | tee "${RESULT_DIR}/distinct_red_dbo_${mode}.log"
done
