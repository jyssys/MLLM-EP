#!/usr/bin/env bash
set -euo pipefail

dataset_name="${1:?dataset name required}"
repeat="${2:-1}"
repo=/home/esjung/MLLM-EP-refinegemm
task_root="$repo/poc_refinegemm"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-discovery
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$repo/poc_llada2_flash_ep/data/bounded_eval_32/${dataset_name}_32.json"
tag="clean_gpu4_7_${dataset_name}_ep4_b32_mini32_r${repeat}_g32"
out="$task_root/results/clean_gpu4_7/$dataset_name/r${repeat}"
log="$task_root/results/clean_gpu4_7/${tag}.log"
gpu_log="$task_root/results/clean_gpu4_7/${tag}_gpu.csv"
mkdir -p "$out" "$task_root/results/clean_gpu4_7"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"

expected=(
  GPU-6076e2f2-5b63-3761-5586-56ceb7df8139
  GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730
  GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28
  GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b
)
mapfile -t observed < <(nvidia-smi -i 4,5,6,7 --query-gpu=uuid --format=csv,noheader)
for index in 0 1 2 3; do
  [[ "${observed[$index]}" == "${expected[$index]}" ]] || {
    echo "physical GPU UUID mismatch at index $((index+4))" >&2
    exit 2
  }
done
for index in 4 5 6 7; do
  mapfile -t pids < <(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
  if ((${#pids[@]})); then
    echo "GPU $index conflict; refusing launch" >&2
    ps -o user,pid,ppid,etime,args -p "$(IFS=,; echo "${pids[*]}")" >&2 || true
    exit 3
  fi
done

unset LLADA_DENOISE_TRACE LLADA_WAVE_TRACE LLADA_DISCOVERY_TRACE
unset LLADA_EP_TRACE_DIR LLADA_EP_TRACE_LAYERS LLADA_EP_TRACE_ROUTES
unset LLADA_EP_SHAPE_TRACE_DIR LLADA_EP_SHAPE_TRACE_LAYERS
unset LLADA_BLOCK_TRACE_DIR LLADA_BLOCK_TRACE_LAYERS LLADA_EP_TEMPORAL_TRACE
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

nvidia-smi -i 4,5,6,7 \
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
