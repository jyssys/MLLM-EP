#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEEPEP_VENV=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
DEEPEP_WORKSPACE=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
RESULT_DIR=${1:?usage: run_operator_replay.sh RESULT_DIR}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
CAPTURE=${CAPTURE:-/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt}

export NVSHMEM_DIR="${DEEPEP_WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/poc_flashvep/deepep_revalidation/hooks:${REPO_ROOT}:${PYTHONPATH:-}"
export FLASHVEP_DEEPEP_PROOF_DIR="${RESULT_DIR}/backend_proof/operator_replay"
export FLASHVEP_DEEPEP_REPLAY_RESULT_DIR="${RESULT_DIR}/operator_replay"
export FLASHVEP_DEEPEP_CAPTURE_PATH="${CAPTURE}"
export FLASHVEP_DEEPEP_REPLAY_LAYER=24
export FLASHVEP_DEEPEP_REPLAY_WARMUPS=${WARMUPS:-5}
export FLASHVEP_DEEPEP_REPLAY_ITERATIONS=${ITERATIONS:-20}
export FLASHVEP_DEEPEP_REPLAY_BATCHES=${BATCHES:-32,64,128}
export FLASHVEP_DEEPEP_REPLAY_SMS=${SMS_VALUES:-20,16,12,8,4}
mkdir -p "${FLASHVEP_DEEPEP_PROOF_DIR}" "${FLASHVEP_DEEPEP_REPLAY_RESULT_DIR}"

"${DEEPEP_VENV}/bin/python" \
  "${REPO_ROOT}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
  --model-path "${MODEL}" \
  --output "${RESULT_DIR}/operator_trigger.json" \
  --all2all-backend deepep_high_throughput \
  --request-counts 1 --warmups 0 --iterations 1 \
  2>&1 | tee "${RESULT_DIR}/operator_replay.log"
