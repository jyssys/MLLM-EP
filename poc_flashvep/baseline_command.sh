#!/usr/bin/env bash
set -euo pipefail

# Historical archived baseline (preserved for provenance; DO NOT execute under
# the current policy because it exposes forbidden physical GPU 0):
#
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# VLLM_WORKER_MULTIPROC_METHOD=spawn \
# python3 scripts/vllm_ep_sanity.py \
#   --kv-cache-memory-bytes 1073741824 \
#   --max-model-len 512 \
#   --max-num-batched-tokens 512 \
#   --output outputs/vllm_ep_sanity.json

# Current seven-GPU reproduction. vLLM 0.20.0 rejects TP=7 because the model
# has 32 attention heads. This command is intentionally retained as the exact
# smoke-blocker reproducer; it does not claim to be a valid performance run.
readonly POC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_ID="smoke_tp7_$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_PATH="${POC_ROOT}/poc_flashvep/results/baseline/${RUN_ID}.json"

if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "Refusing to overwrite existing result: ${OUTPUT_PATH}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${POC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${POC_ROOT}"
exec conda run -n flashvep-poc python scripts/vllm_ep_sanity.py \
  --model-path Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 7 \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --output "${OUTPUT_PATH}"
