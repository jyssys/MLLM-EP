#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULT_DIR=${1:?usage: run_nsight_best.sh RESULT_DIR}
NSIGHT_RUN_DIR="${RESULT_DIR}/nsight_best_b128_sm8"
TRACE_DIR="${RESULT_DIR}/large_local_artifacts"
mkdir -p "${NSIGHT_RUN_DIR}" "${TRACE_DIR}"

BATCHES=128 SMS_VALUES=8 WARMUPS=1 ITERATIONS=3 \
  nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --output="${TRACE_DIR}/deepep_b128_sm8" \
    "${REPO_ROOT}/poc_flashvep/deepep_revalidation/run_operator_replay.sh" \
    "${NSIGHT_RUN_DIR}" \
    2>&1 | tee "${RESULT_DIR}/nsight_best.log"

nsys stats \
  --force-export=true \
  --report=cuda_gpu_kern_sum \
  --format=csv \
  --output="${RESULT_DIR}/nsight_kernel_summary" \
  "${TRACE_DIR}/deepep_b128_sm8.nsys-rep"
