#!/usr/bin/env bash
set -euo pipefail

# The experiment is intentionally restricted to physical GPUs 1--4.
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
WORKSPACE=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
ROUTES=${ROUTES:-${ROOT}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609}
CAPTURE=${CAPTURE:-/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt}
RESULT=${1:?usage: run_gpu.sh RESULT_DIR}

if [[ -e "$RESULT" ]]; then
  echo "refusing to overwrite $RESULT" >&2
  exit 2
fi

export NVSHMEM_DIR="${WORKSPACE}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
# Prepare with the system Python before publishing the worker-only startup
# hook.  The hook imports torch/vLLM and must not run in this analysis step.
python -m poc_flashvep.modality_aware_moe_granularity.prepare \
  --output "$RESULT" --route-root "$ROUTES"

export PYTHONPATH="${ROOT}/poc_flashvep/modality_aware_moe_granularity/hooks:${ROOT}:${PYTHONPATH:-}"
export FLASHVEP_GRANULARITY_RESULT_DIR="$RESULT"
export FLASHVEP_GRANULARITY_CAPTURE="$CAPTURE"
export FLASHVEP_GRANULARITY_WARMUPS=${WARMUPS:-3}
export FLASHVEP_GRANULARITY_ITERATIONS=${ITERATIONS:-20}
export FLASHVEP_CONFIGURED_ALL2ALL_BACKEND=deepep_high_throughput
export FLASHVEP_CONFIGURED_DBO=false

# The stock backend matrix driver creates the validated TP2/DP2/EP4 vLLM
# worker topology and submits one real image request.  Our startup hook only
# adds bounded operator replay at layers 4/24/44 and returns stock results.
"$VENV/bin/python" "$ROOT/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
  --model-path "$MODEL" --output "$RESULT/trigger.json" \
  --all2all-backend deepep_high_throughput \
  --request-counts 1 --warmups 0 --iterations 1 \
  --modality vision --image-size 448 --text-target-tokens 790 \
  --max-tokens 1 --kv-cache-memory-bytes 1073741824 \
  --timeout-seconds 14400 2>&1 | tee "$RESULT/run.log"
