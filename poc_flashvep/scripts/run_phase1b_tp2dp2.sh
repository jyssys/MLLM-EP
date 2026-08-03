#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${FLASHVEP_PYTHON:-/home/esjung/anaconda3/envs/flashvep-poc/bin/python}"
model_path="${FLASHVEP_MODEL_PATH:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}"
run_id="phase1b_tp2dp2_vision896_$(date +%Y%m%d_%H%M%S)"
result_dir="${1:-${repo_root}/poc_flashvep/results/${run_id}}"

if [[ -e "${result_dir}" ]]; then
  echo "Refusing to overwrite existing result directory: ${result_dir}" >&2
  exit 1
fi
mkdir -p "${result_dir}"

export PYTHONPATH="${repo_root}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export FLASHVEP_PHYSICAL_GPUS=4,5,6,7

common=(
  --model-path "${model_path}"
  --warmups 5
  --iterations 20
  --moe-backend auto
  --image-size 896
  --max-model-len 1024
  --max-num-batched-tokens 1024
  --timeout-seconds 1500
)

"${python_bin}" "${repo_root}/poc_flashvep/scripts/phase1b_tp2dp2.py" \
  "${common[@]}" \
  --output "${result_dir}/baseline_requests.json"

export FLASHVEP_PHASE1B_RUN_ID="${run_id}_profile"
export FLASHVEP_PHASE1B_AUDIT_JSONL="${result_dir}/audit.jsonl"
export FLASHVEP_PHASE1B_PROFILE_JSONL="${result_dir}/stage_events.jsonl"
export FLASHVEP_PHASE1B_LAYERS=0,12,24,36,47
export FLASHVEP_PHASE1B_SKIP_LAYER_CALLS=162
export FLASHVEP_PHASE1B_LAYER_CALL_STRIDE=32
export FLASHVEP_PHASE1B_MEASURE_LAYER_CALLS=20
export FLASHVEP_PHASE1B_VISION_START=4
export FLASHVEP_PHASE1B_VISION_END=788
export FLASHVEP_PHASE1B_REAL_PROMPT_TOKENS=799
export FLASHVEP_PHASE1B_REAL_DP_TP_CHUNKS=2
export FLASHVEP_PHASE1B_CAPTURE_LAYER=24
export FLASHVEP_PHASE1B_MICROBENCH=1
export FLASHVEP_PHASE1B_MICROBENCH_WARMUPS=20
export FLASHVEP_PHASE1B_MICROBENCH_ITERATIONS=100

"${python_bin}" "${repo_root}/poc_flashvep/scripts/phase1b_tp2dp2.py" \
  "${common[@]}" \
  --microbenchmark \
  --output "${result_dir}/profile_requests.json"

"${python_bin}" "${repo_root}/poc_flashvep/scripts/analyze_phase1b_tp2dp2.py" \
  --events "${result_dir}/stage_events.jsonl" \
  --baseline "${result_dir}/baseline_requests.json" \
  --profile "${result_dir}/profile_requests.json" \
  --output "${result_dir}/analysis.json" \
  --tiles 400

echo "Phase 1b results: ${result_dir}"
