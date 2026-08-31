#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,2,3,4 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-/home/esjung/.venvs/flashvep-deepep-v020}; WORKSPACE=${WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}; MODEL=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}; RESULT=${1:?usage: run_gpu.sh RESULT_DIR}
export PATH="$VENV/bin:$PATH" NVSHMEM_DIR="$WORKSPACE/nvshmem"; export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/poc_flashvep/true_gpu_chunk_oracle_poc/hooks:$REPO_ROOT:${PYTHONPATH:-}"
export TRUE_REPLAY_DIR="${TRUE_REPLAY_DIR:-$RESULT/replay}" TRUE_CANDIDATES="${TRUE_CANDIDATES:-$RESULT/candidates.json}" TRUE_CAPTURE="/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt" TRUE_SHORT_ROUTE_DIR="$REPO_ROOT/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609" TRUE_LONG_ROUTE_DIR="$REPO_ROOT/poc_flashvep/deepep_revalidation/results/chunk_oracle_gpu_scale_validation_20260831_223000" TRUE_WARMUPS="${WARMUPS:-5}" TRUE_ITERATIONS="${ITERATIONS:-20}" TRUE_B_MODE="${TRUE_B_MODE:-}" TRUE_INTERVALS="${TRUE_INTERVALS:-$RESULT/stage_b_intervals.json}" TRUE_B_REPLAY_DIR="${TRUE_B_REPLAY_DIR:-$RESULT/stage_b_cost}" TRUE_CUTS="${TRUE_CUTS:-$RESULT/stage_b_cuts.json}"
mkdir -p "$RESULT/replay"
"$VENV/bin/python" "$REPO_ROOT/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" --model-path "$MODEL" --output "$RESULT/trigger.json" --all2all-backend deepep_high_throughput --request-counts 1 --warmups 0 --iterations 1 --modality vision --image-size 896 --text-target-tokens 790 --max-tokens 1 --kv-cache-memory-bytes 1073741824 --timeout-seconds 14400 2>&1 | tee "$RESULT/run.log"
