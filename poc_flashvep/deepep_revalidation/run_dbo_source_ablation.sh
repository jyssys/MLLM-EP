#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT=${1:?usage: run_dbo_source_ablation.sh RESULT_ROOT}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MATRIX="${REPO_ROOT}/poc_flashvep/deepep_revalidation/run_dbo_correctness_matrix.sh"
SET_MODE="${REPO_ROOT}/poc_flashvep/deepep_revalidation/set_vllm_source_fix_mode.sh"

unset FLASHVEP_DBO_CORRECTNESS_FIX || true
export ENABLE_TRACE=1
export WARMUPS=${WARMUPS:-1}
export ITERATIONS=${ITERATIONS:-3}
export CASES="text:on vision:on"

for mode in none attention deepstack both; do
  "${SET_MODE}" "${mode}"
  "${MATRIX}" "${RESULT_ROOT}/${mode}" baseline
done
