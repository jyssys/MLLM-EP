#!/usr/bin/env bash
set -euo pipefail

topology="$1"
submitted_batch="$2"
mini_batch_size="$3"
repeat="$4"
generation="${5:-32}"

root=/home/esjung/MLLM-EP-raws
task_root="$root/poc_refinement_wave_sizing"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-poc
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$root/poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32.json"
tag="${topology}_b${submitted_batch}_mini${mini_batch_size}_r${repeat}_g${generation}"
out="$task_root/results/clean/$topology/b${submitted_batch}/mini${mini_batch_size}/r${repeat}"
log="$task_root/logs/clean_${tag}.log"
gpu_log="$task_root/logs/gpu_${tag}.csv"
mkdir -p "$out" "$task_root/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3
active_pids="$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 0-3 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi
unset LLADA_DENOISE_TRACE LLADA_WAVE_TRACE LLADA_EP_TRACE_DIR
unset LLADA_EP_TRACE_ROUTES LLADA_EP_SHAPE_TRACE_DIR LLADA_BLOCK_TRACE_DIR
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

common=(
  "$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py"
  --model_name "$model"
  --dataset "$dataset"
  --gpu 0,1,2,3
  --batch_size "$submitted_batch"
  --gen_len "$generation"
  --block_length 32
  --threshold 0.9
  --config 42
  --model_type flash
  --mini_batch_size "$mini_batch_size"
  --output_dir "$out"
  --exp_name "$tag"
)

if [[ "$topology" == "ep4" ]]; then
  "${common[@]}" --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
    2>&1 | tee "$log"
elif [[ "$topology" == "tp4" ]]; then
  "${common[@]}" --ep_size 1 --moe_a2a_backend none 2>&1 | tee "$log"
else
  echo "unsupported topology: $topology" >&2
  exit 2
fi
