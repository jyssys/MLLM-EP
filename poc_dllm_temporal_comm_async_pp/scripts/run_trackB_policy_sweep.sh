#!/usr/bin/env bash
set -euo pipefail

dataset_name="$1"
repeat="$2"
result_root="$3"
plan="$4"

repo=/home/esjung/MLLM-EP-temporal-comm-async-pp
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-temporal-comm-async
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="trackB_policy_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$result_root/trackB/policy_sweep/$dataset_name/r${repeat}"
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

unset LLADA_TEMPORAL_COMM_TRACE_DIR LLADA_PP_BOUNDARY_TRACE_DIR
unset LLADA_BLOCK_TRACE_DIR LLADA_EP_TRACE_DIR LLADA_DISCOVERY_TRACE
export LLADA_WAVE_TRACE=1
export LLADA_ASYNC_POLICY_TRACE_DIR="$out"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size 32 --mini_batch_size 32 --gen_len 32 --block_length 32 \
  --threshold 0.9 --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" --intervention_plan "$plan" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  >"$log" 2>&1

rg -q '^Forward:' "$log"
test -s "$out/policy_baseline.json"
rg '^Forward:' "$log" | tail -1
