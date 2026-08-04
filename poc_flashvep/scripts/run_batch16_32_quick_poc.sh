#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${FLASHVEP_PYTHON:-/home/esjung/anaconda3/envs/flashvep-poc/bin/python}"
model_path="${FLASHVEP_MODEL_PATH:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}"
run_id="batch16_32_quick_poc_$(date +%Y%m%d_%H%M%S)"
result_dir="${1:-${repo_root}/poc_flashvep/results/${run_id}}"
batch1_analysis="${repo_root}/poc_flashvep/results/phase1b_tp2dp2_vision896/analysis_final.json"

if [[ -e "${result_dir}" ]]; then
  echo "Refusing to overwrite existing result directory: ${result_dir}" >&2
  exit 1
fi
mkdir -p "${result_dir}"

export PYTHONPATH="${repo_root}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export FLASHVEP_PHYSICAL_GPUS=4,5,6,7

run_batch() {
  local batch_size="$1"
  local token_budget="$2"
  local tiles=$((batch_size * 799 / 4))
  local batch_dir="${result_dir}/batch${batch_size}"
  mkdir -p "${batch_dir}"

  local common=(
    --model-path "${model_path}"
    --global-batch-size "${batch_size}"
    --warmups 3
    --iterations 8
    --moe-backend auto
    --image-size 896
    --max-model-len 1024
    --max-num-batched-tokens "${token_budget}"
    --timeout-seconds 2400
  )

  unset FLASHVEP_PHASE1B_RUN_ID
  unset FLASHVEP_PHASE1B_AUDIT_JSONL
  unset FLASHVEP_PHASE1B_PROFILE_JSONL
  unset FLASHVEP_PHASE1B_GLOBAL_BATCH_SIZE
  unset FLASHVEP_PHASE1B_MICROBENCH
  unset FLASHVEP_PHASE1B_CAPTURE_LAYER

  "${python_bin}" "${repo_root}/poc_flashvep/scripts/phase1b_tp2dp2.py" \
    "${common[@]}" \
    --output "${batch_dir}/baseline_requests.json" \
    2>&1 | tee "${batch_dir}/baseline.log"

  export FLASHVEP_PHASE1B_RUN_ID="${run_id}_batch${batch_size}_profile"
  export FLASHVEP_PHASE1B_AUDIT_JSONL="${batch_dir}/audit.jsonl"
  export FLASHVEP_PHASE1B_PROFILE_JSONL="${batch_dir}/stage_events.jsonl"
  export FLASHVEP_PHASE1B_LAYERS=12,24,36
  export FLASHVEP_PHASE1B_SKIP_LAYER_CALLS=98
  export FLASHVEP_PHASE1B_LAYER_CALL_STRIDE=32
  export FLASHVEP_PHASE1B_LAYER_CALL_OFFSETS=0,1,2
  export FLASHVEP_PHASE1B_MEASURE_LAYER_CALLS=8
  export FLASHVEP_PHASE1B_REAL_PROMPT_TOKENS=799
  export FLASHVEP_PHASE1B_REAL_DP_TP_CHUNKS=2
  export FLASHVEP_PHASE1B_GLOBAL_BATCH_SIZE="${batch_size}"

  "${python_bin}" "${repo_root}/poc_flashvep/scripts/phase1b_tp2dp2.py" \
    "${common[@]}" \
    --output "${batch_dir}/profile_requests.json" \
    2>&1 | tee "${batch_dir}/profile.log"

  "${python_bin}" \
    "${repo_root}/poc_flashvep/scripts/analyze_batch16_32_quick_poc.py" \
    --batch-size "${batch_size}" \
    --events "${batch_dir}/stage_events.jsonl" \
    --audit "${batch_dir}/audit.jsonl" \
    --baseline "${batch_dir}/baseline_requests.json" \
    --profile "${batch_dir}/profile_requests.json" \
    --batch1-analysis "${batch1_analysis}" \
    --output "${batch_dir}/analysis.json" \
    --layers 12,24,36 \
    --iterations 8

  "${python_bin}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d["profiler_overhead_fraction"] < 0.20 else 20)' \
    "${batch_dir}/analysis.json"

  echo "Batch ${batch_size} complete: ${batch_dir}; first-token tiles=${tiles}"
}

run_batch 16 8192
run_batch 32 16384

echo "Batch 16/32 Quick PoC results: ${result_dir}"
