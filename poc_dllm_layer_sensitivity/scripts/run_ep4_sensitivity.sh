#!/usr/bin/env bash
set -euo pipefail

kind="$1"
dataset_name="$2"
repeat="$3"
plan="${4:-}"

repo=/home/esjung/MLLM-EP-layer-sensitivity
task_root="$repo/poc_dllm_layer_sensitivity"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-layer-sensitivity
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="${kind}_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$task_root/results/raw/$kind/$dataset_name/r${repeat}"
capture="$task_root/results/captures/$kind/$dataset_name/r${repeat}"
log="$task_root/logs/${tag}.log"
mkdir -p "$out" "$capture" "$task_root/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"

active_pids="$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 0-3 conflict; refusing launch" >&2
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
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
all_layers="$(seq -s, 0 31)"

extra=()
case "$kind" in
  clean)
    ;;
  stability)
    export LLADA_DENOISE_TRACE=1
    export LLADA_WAVE_TRACE=1
    export LLADA_DISCOVERY_TRACE=1
    export LLADA_LAYER_SENSITIVITY_TRACE_DIR="$out"
    export LLADA_EP_SHAPE_TRACE_DIR="$out"
    export LLADA_EP_SHAPE_TRACE_LAYERS="$all_layers"
    export LLADA_EP_TRACE_DIR="$out"
    export LLADA_EP_TRACE_LAYERS="$all_layers"
    export LLADA_BLOCK_TRACE_DIR="$out"
    export LLADA_BLOCK_TRACE_LAYERS="$all_layers"
    ;;
  causal_*)
    if [[ -z "$plan" ]]; then
      echo "causal run requires a policy-plan path" >&2
      exit 2
    fi
    export LLADA_DENOISE_TRACE=1
    export LLADA_SENSITIVITY_CAPTURE_DIR="$capture"
    extra+=(--intervention_plan "$plan")
    ;;
  *)
    echo "unknown run kind: $kind" >&2
    exit 2
    ;;
esac

start_epoch="$(date +%s)"
"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size 32 --mini_batch_size 32 --gen_len 32 --block_length 32 \
  --threshold 0.9 --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  "${extra[@]}" >"$log" 2>&1
end_epoch="$(date +%s)"

rg -q '^Forward:' "$log"
rg '^Forward:' "$log" | tail -1
echo "$start_epoch,$end_epoch,$((end_epoch-start_epoch)),0-3,$kind,$dataset_name,ep4_b32_mini32_g32,$log" \
  >> "$task_root/GPU_TIME_LOG.raw.csv"
