#!/usr/bin/env bash
set -euo pipefail

readonly POC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly RUN_ID="tp4_phase1_vision896_$(date +%Y%m%d_%H%M%S)"
readonly RUN_DIR="${POC_ROOT}/poc_flashvep/results/${RUN_ID}"
readonly PYTHON_BIN="${FLASHVEP_PYTHON_BIN:-/home/esjung/anaconda3/envs/flashvep-poc/bin/python}"
readonly MODEL_PATH="${FLASHVEP_MODEL_PATH:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${POC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${POC_ROOT}"

"${PYTHON_BIN}" poc_flashvep/scripts/profile_tp4.py \
  --model-path "${MODEL_PATH}" \
  --output "${RUN_DIR}/baseline_requests.json" \
  --warmups 5 --iterations 20 --tensor-parallel-size 4 \
  --moe-backend triton --image-size 896 \
  --max-model-len 1024 --max-num-batched-tokens 1024

env \
  FLASHVEP_PROFILE_JSONL="${RUN_DIR}/stages.jsonl" \
  FLASHVEP_RUN_ID="${RUN_ID}" \
  FLASHVEP_SKIP_LAYER_CALLS=8 \
  FLASHVEP_MEASURE_LAYER_CALLS=20 \
  FLASHVEP_PHYSICAL_GPUS=4,5,6,7 \
  "${PYTHON_BIN}" poc_flashvep/scripts/profile_tp4.py \
    --model-path "${MODEL_PATH}" \
    --output "${RUN_DIR}/profile_requests.json" \
    --warmups 5 --iterations 20 --tensor-parallel-size 4 \
    --moe-backend triton --image-size 896 \
    --max-model-len 1024 --max-num-batched-tokens 1024

env \
  FLASHVEP_PROFILE_JSONL="${RUN_DIR}/lean_stages.jsonl" \
  FLASHVEP_PROFILE_STAGES=decoder_layer,router_topk \
  FLASHVEP_RUN_ID="${RUN_ID}_lean" \
  FLASHVEP_SKIP_LAYER_CALLS=8 \
  FLASHVEP_MEASURE_LAYER_CALLS=20 \
  FLASHVEP_PHYSICAL_GPUS=4,5,6,7 \
  "${PYTHON_BIN}" poc_flashvep/scripts/profile_tp4.py \
    --model-path "${MODEL_PATH}" \
    --output "${RUN_DIR}/lean_requests.json" \
    --warmups 5 --iterations 20 --tensor-parallel-size 4 \
    --moe-backend triton --image-size 896 \
    --max-model-len 1024 --max-num-batched-tokens 1024

"${PYTHON_BIN}" poc_flashvep/scripts/analyze_tp4_profile.py \
  --stages "${RUN_DIR}/stages.jsonl" \
  --requests "${RUN_DIR}/profile_requests.json" \
  --output-dir "${RUN_DIR}/analysis"

echo "Phase 1 results: ${RUN_DIR}"
