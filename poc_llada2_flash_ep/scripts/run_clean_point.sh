#!/usr/bin/env bash
set -euo pipefail

topology="$1"
batch="$2"
repeat="$3"
mode="${4:-dynamic4}"
generation="${5:-32}"
if [[ "$mode" =~ ^dynamic([0-9]+)$ ]]; then
  mini_batch_size="${BASH_REMATCH[1]}"
elif [[ "$mode" == "naive" ]]; then
  mini_batch_size="$batch"
else
  echo "unknown batching mode: $mode" >&2
  exit 2
fi

root=/home/esjung/MLLM-EP-llada2-flash-100b
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-poc
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$root/poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32.json"
out="$root/poc_llada2_flash_ep/results/clean/${topology}/${mode}/b${batch}/r${repeat}"
log="$root/poc_llada2_flash_ep/logs/clean_${topology}_${mode}_b${batch}_r${repeat}.log"
mkdir -p "$out" "$(dirname "$log")"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits | sed -n '1,4p'

# Keep the launch literal below so the GPU/model scope is auditable in logs.
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
  --exp_name "${topology}_${mode}_b${batch}_r${repeat}"
)

if [[ "$mode" == "naive" ]]; then
  common+=(--use_naive_batching)
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ "$topology" == "tp4" ]]; then
  "${common[@]}" --ep_size 1 --moe_a2a_backend none 2>&1 | tee "$log"
elif [[ "$topology" == "hybrid_ep2" ]]; then
  "${common[@]}" --ep_size 2 --moe_a2a_backend none 2>&1 | tee "$log"
elif [[ "$topology" == "ep4" ]]; then
  export SGL_ENABLE_JIT_DEEPGEMM=false
  export SGLANG_ENABLE_JIT_DEEPGEMM=false
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
  "${common[@]}" --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal 2>&1 | tee "$log"
else
  echo "unknown topology: $topology" >&2
  exit 2
fi
