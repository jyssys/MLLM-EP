#!/usr/bin/env bash
set -euo pipefail

kind="$1"
dataset_name="$2"
repeat="$3"
plan="${4:-}"

repo=/home/esjung/MLLM-EP-dormant-live
task_root="$repo/poc_dormant_live_ep_refresh"
campaign="$task_root/results/dormant_live_ep_refresh_20260914_152829"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-dormant-live
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="${kind}_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$campaign/raw/$kind/$dataset_name/r${repeat}"
log="$campaign/logs/${tag}.log"
mkdir -p "$out" "$campaign/logs"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"

# This check uses physical indices before CUDA remaps them.  Never terminate
# an unknown process here; the caller must identify and stop task-owned burn.
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | sort -u || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi

unset LLADA_DENOISE_TRACE LLADA_WAVE_TRACE LLADA_DISCOVERY_TRACE
unset LLADA_EP_TRACE_DIR LLADA_EP_TRACE_LAYERS LLADA_EP_TRACE_ROUTES
unset LLADA_EP_SHAPE_TRACE_DIR LLADA_EP_SHAPE_TRACE_LAYERS
unset LLADA_BLOCK_TRACE_DIR LLADA_BLOCK_TRACE_LAYERS LLADA_EP_TEMPORAL_TRACE
unset LLADA_LAYER_SENSITIVITY_TRACE_DIR LLADA_SENSITIVITY_CAPTURE_DIR
unset LLADA_INTERVENTION_MODE LLADA_INTERVENTION_LAYERS
unset LLADA_INTERVENTION_PHASES LLADA_INTERVENTION_WAVES
unset LLADA_DORMANT_DETAIL_TRACE LLADA_DORMANT_INTERVENTION_TRACE_DIR
unset LLADA_DORMANT_ACCEPTANCE_PLAN LLADA_DORMANT_HORIZON
unset LLADA_DORMANT_REFRESH_PERIOD LLADA_DORMANT_ROUTE_TRIGGER
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
all_layers="$(seq -s, 0 31)"

extra=()
case "$kind" in
  clean)
    ;;
  timing)
    export LLADA_DENOISE_TRACE=1
    export LLADA_WAVE_TRACE=1
    export LLADA_EP_TRACE_DIR="$out"
    export LLADA_EP_TRACE_LAYERS="$all_layers"
    ;;
  shape)
    export LLADA_DENOISE_TRACE=1
    export LLADA_DISCOVERY_TRACE=1
    export LLADA_EP_SHAPE_TRACE_DIR="$out"
    export LLADA_EP_SHAPE_TRACE_LAYERS="$all_layers"
    ;;
  detail)
    export LLADA_DENOISE_TRACE=1
    export LLADA_DISCOVERY_TRACE=1
    export LLADA_EP_SHAPE_TRACE_DIR="$out"
    export LLADA_EP_SHAPE_TRACE_LAYERS="1,8,16,24,31"
    export LLADA_DORMANT_DETAIL_TRACE=1
    ;;
  causal)
    if [[ -z "$plan" ]]; then
      echo "causal run requires a policy-plan path" >&2
      exit 2
    fi
    export LLADA_DENOISE_TRACE=1
    export LLADA_DISCOVERY_TRACE=1
    export LLADA_DORMANT_INTERVENTION_TRACE_DIR="$out"
    extra+=(--intervention_plan "$plan")
    ;;
  *)
    echo "unknown run kind: $kind" >&2
    exit 2
    ;;
esac

start_epoch="$(date +%s)"
start_iso="$(date -Iseconds)"
set +e
"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size 32 --mini_batch_size 32 --gen_len 32 --block_length 32 \
  --threshold 0.9 --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  "${extra[@]}" >"$log" 2>&1
rc=$?
set -e
end_epoch="$(date +%s)"
end_iso="$(date -Iseconds)"
status=failed
if [[ "$rc" -eq 0 ]] && rg -q '^Forward:' "$log"; then
  status=ok
  rg '^Forward:' "$log" | tail -1
fi
echo "$start_iso,$end_iso,$((end_epoch-start_epoch)),4-7,$kind,$dataset_name,ep4_b32_mini32_g32,$status,$log" \
  >> "$task_root/GPU_TIME_LOG.csv"
exit "$rc"
