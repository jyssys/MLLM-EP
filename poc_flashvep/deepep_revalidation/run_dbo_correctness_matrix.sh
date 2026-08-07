#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
RESULT_DIR=${1:?usage: run_dbo_correctness_matrix.sh RESULT_DIR [baseline|fixed]}
MODE=${2:-baseline}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
WARMUPS=${WARMUPS:-1}
ITERATIONS=${ITERATIONS:-3}
ENABLE_TRACE=${ENABLE_TRACE:-1}

export NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/deepep_revalidation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
if [[ "${MODE}" == "fixed" ]]; then
  export FLASHVEP_DBO_CORRECTNESS_FIX=1
else
  unset FLASHVEP_DBO_CORRECTNESS_FIX || true
fi

mkdir -p "${RESULT_DIR}"

run_case() {
  local modality=$1
  local dbo=$2
  local name="${modality}_dbo_${dbo}"
  local dbo_args=()
  if [[ "${dbo}" == "on" ]]; then
    dbo_args+=(--enable-dbo)
  fi
  local expected_token=2132
  if [[ "${modality}" == "vision" ]]; then
    expected_token=1986
  fi
  if [[ "${ENABLE_TRACE}" == "1" ]]; then
    export FLASHVEP_DBO_CORRECTNESS_TRACE_DIR="${RESULT_DIR}/traces/${name}"
    mkdir -p "${FLASHVEP_DBO_CORRECTNESS_TRACE_DIR}"
  else
    unset FLASHVEP_DBO_CORRECTNESS_TRACE_DIR || true
  fi
  "${DEEPEP_VENV}/bin/python" \
    "${REPO_ROOT}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
    --model-path "${MODEL}" \
    --output "${RESULT_DIR}/${name}.json" \
    --all2all-backend deepep_high_throughput \
    --modality "${modality}" \
    --request-counts 2 4 8 \
    --expected-output-token "${expected_token}" \
    --warmups "${WARMUPS}" \
    --iterations "${ITERATIONS}" \
    --allow-correctness-failure \
    "${dbo_args[@]}" \
    2>&1 | tee "${RESULT_DIR}/${name}.log"
}

CASES=${CASES:-"text:off text:on vision:off vision:on"}
for case_spec in ${CASES}; do
  run_case "${case_spec%%:*}" "${case_spec##*:}"
done
