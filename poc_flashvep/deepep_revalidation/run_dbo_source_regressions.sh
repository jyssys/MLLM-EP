#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
RESULT_DIR=${1:?usage: run_dbo_source_regressions.sh RESULT_DIR}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}

export NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/deepep_revalidation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
unset FLASHVEP_DBO_CORRECTNESS_FIX FLASHVEP_DBO_CORRECTNESS_TRACE_DIR || true
mkdir -p "${RESULT_DIR}"

run_case() {
  local scenario=$1
  local dbo=$2
  local expected=$3
  local dbo_args=()
  if [[ "${dbo}" == "on" ]]; then
    dbo_args+=(--enable-dbo)
  fi
  "${DEEPEP_VENV}/bin/python" \
    "${REPO_ROOT}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
    --model-path "${MODEL}" \
    --output "${RESULT_DIR}/${scenario}_dbo_${dbo}.json" \
    --all2all-backend deepep_high_throughput \
    --scenario "${scenario}" \
    --request-counts 4 \
    --expected-output-tokens ${expected} \
    --warmups 1 \
    --iterations 3 \
    "${dbo_args[@]}" \
    2>&1 | tee "${RESULT_DIR}/${scenario}_dbo_${dbo}.log"
}

# These tokens are established by the matching DBO-off runs and checked again
# for every request slot on both DP ranks.
run_case mixed_length off "2132 2132"
run_case mixed_length on "2132 2132"
run_case mixed_modality off "2132 1986"
run_case mixed_modality on "2132 1986"
