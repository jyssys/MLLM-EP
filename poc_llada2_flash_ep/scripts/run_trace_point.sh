#!/usr/bin/env bash
set -euo pipefail

topology="$1"
batch="$2"
tag="$3"
routes="${4:-1}"
generation="${5:-32}"
mini_batch_size="${6:-4}"

root=/home/esjung/MLLM-EP-llada2-flash-100b
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-poc
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
if (( batch > 8 )); then
  dataset="$root/poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32.json"
else
  dataset="$root/poc_llada2_flash_ep/data/bounded_eval/gsm8k_8.json"
fi
out="$root/poc_llada2_flash_ep/results/trace/${topology}/b${batch}/${tag}"
log="$root/poc_llada2_flash_ep/logs/trace_${topology}_b${batch}_${tag}.log"
mkdir -p "$out" "$(dirname "$log")"

# Query only the physical allow-list. This is also retained in the run log.
nvidia-smi -i 0,1,2,3 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LLADA_BLOCK_TRACE_DIR="$out"
if [[ "$routes" == "block_only" ]]; then
  unset LLADA_EP_TRACE_DIR LLADA_EP_TRACE_ROUTES
else
  export LLADA_EP_TRACE_DIR="$out"
  export LLADA_EP_TRACE_ROUTES="$routes"
fi
export LLADA_DENOISE_TRACE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

common=(
  "$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py"
  --model_name "$model"
  --dataset "$dataset"
  --gpu 0,1,2,3
  --batch_size "$batch"
  --gen_len "$generation"
  --block_length 32
  --threshold 0.9
  --config 42
  --model_type flash
  --mini_batch_size "$mini_batch_size"
  --output_dir "$out"
  --exp_name "trace_${topology}_b${batch}_${tag}"
)

if [[ "$topology" == "tp4" ]]; then
  "${common[@]}" --ep_size 1 --moe_a2a_backend none 2>&1 | tee "$log"
elif [[ "$topology" == "ep4" ]]; then
  "${common[@]}" --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal 2>&1 | tee "$log"
else
  echo "unsupported trace topology: $topology" >&2
  exit 2
fi
