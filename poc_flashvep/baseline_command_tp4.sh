#!/usr/bin/env bash
set -euo pipefail

# Authorized TP4/effective-EP4 baseline on physical GPUs 4-7. The archived
# TP7 failure reproducer remains unchanged in baseline_command.sh.
readonly POC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_ID="tp4_baseline_vision896_$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_PATH="${POC_ROOT}/poc_flashvep/results/${RUN_ID}/requests.json"
readonly PYTHON_BIN="${FLASHVEP_PYTHON_BIN:-/home/esjung/anaconda3/envs/flashvep-poc/bin/python}"
readonly MODEL_PATH="${FLASHVEP_MODEL_PATH:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${POC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${POC_ROOT}"
exec "${PYTHON_BIN}" poc_flashvep/scripts/profile_tp4.py \
  --model-path "${MODEL_PATH}" \
  --output "${OUTPUT_PATH}" \
  --warmups 5 \
  --iterations 20 \
  --tensor-parallel-size 4 \
  --moe-backend triton \
  --image-size 896 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024
