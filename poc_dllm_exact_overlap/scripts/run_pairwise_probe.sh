#!/usr/bin/env bash
set -euo pipefail

dataset_name="$1"
target_wave="$2"
repeat="$3"
result_root="${4:-/home/esjung/MLLM-EP-exact-overlap/poc_dllm_exact_overlap/results/exact_overlap_20260913_134622}"

repo=/home/esjung/MLLM-EP-exact-overlap
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-exact-overlap
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="pair_${dataset_name}_wave${target_wave}_layer16_r${repeat}"
out="$result_root/pairwise/$dataset_name/wave${target_wave}/r${repeat}"
log="$result_root/logs/${tag}.log"
gpu_log="$result_root/logs/${tag}_gpu.csv"
mkdir -p "$out" "$result_root/logs"

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
export LLADA_EXACT_OVERLAP_DIR="$out"
export LLADA_EXACT_OVERLAP_LAYER=16
export LLADA_EXACT_OVERLAP_WAVE="$target_wave"
export LLADA_EXACT_OVERLAP_WARMUPS="${LLADA_EXACT_OVERLAP_WARMUPS:-5}"
export LLADA_EXACT_OVERLAP_REPS="${LLADA_EXACT_OVERLAP_REPS:-30}"
export LLADA_DENOISE_TRACE=1
export LLADA_WAVE_TRACE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

nvidia-smi -i 0,1,2,3 \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 100 >"$gpu_log" &
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
  --output_dir "$out/generation" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  >"$log" 2>&1

rg -q '^Forward:' "$log"
test -f "$out/pairwise_rank0.json"
rg '^Forward:' "$log" | tail -1
