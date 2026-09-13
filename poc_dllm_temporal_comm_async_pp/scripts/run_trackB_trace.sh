#!/usr/bin/env bash
set -euo pipefail

dataset_name="$1"
repeat="$2"
result_root="$3"

repo=/home/esjung/MLLM-EP-temporal-comm-async-pp
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-temporal-comm-async
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="trackB_boundary_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$result_root/trackB/boundary_trace/$dataset_name/r${repeat}"
log="$result_root/logs/${tag}.log"
mkdir -p "$out" "$result_root/logs"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi

unset LLADA_TEMPORAL_COMM_TRACE_DIR LLADA_TEMPORAL_COMM_TRACE_LAYERS
unset LLADA_EXACT_OVERLAP_DIR LLADA_EXACT_OVERLAP_LAYER LLADA_EXACT_OVERLAP_WAVE
unset LLADA_DISCOVERY_TRACE LLADA_EP_SHAPE_TRACE_DIR LLADA_LAYER_SENSITIVITY_TRACE_DIR
unset LLADA_ASYNC_BOUNDARY_LAYERS LLADA_ASYNC_WARMUP LLADA_ASYNC_PERIOD LLADA_ASYNC_PHASES
export LLADA_DENOISE_TRACE=1
export LLADA_WAVE_TRACE=1
export LLADA_BLOCK_TRACE_DIR="$out"
export LLADA_BLOCK_TRACE_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
export LLADA_PP_BOUNDARY_TRACE_DIR="$out"
export LLADA_PP_BOUNDARY_TRACE_LAYERS=7,15,23,31
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size 32 --mini_batch_size 32 --gen_len 32 --block_length 32 \
  --threshold 0.9 --config 42 --model_type flash \
  --output_dir "$out/generation" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  >"$log" 2>&1

rg -q '^Forward:' "$log"
test -s "$out/pp_boundary_rank0.jsonl"
rg '^Forward:' "$log" | tail -1
