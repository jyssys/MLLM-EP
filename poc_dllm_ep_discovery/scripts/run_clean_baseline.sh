#!/usr/bin/env bash
set -euo pipefail

dataset_name="$1"
repeat="$2"

repo=/home/esjung/MLLM-EP-discovery
task_root="$repo/poc_dllm_ep_discovery"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-discovery
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="clean_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$task_root/results/clean/$dataset_name/r${repeat}"
log="$task_root/logs/${tag}.log"
gpu_log="$task_root/logs/${tag}_gpu.csv"
mkdir -p "$out" "$task_root/logs"

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
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

nvidia-smi -i 0,1,2,3 \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 >"$gpu_log" &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size 32 --mini_batch_size 32 --gen_len 32 --block_length 32 \
  --threshold 0.9 --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  2>&1 | tee "$log"

rg -q '^Forward:' "$log"
