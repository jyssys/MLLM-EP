#!/usr/bin/env bash
set -euo pipefail

task="$1"
topology="$2"
repeat="$3"
root=/home/esjung/MLLM-EP-llada2-flash-100b
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-flash-poc
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
dataset="$root/poc_llada2_flash_ep/data/bounded_eval_32/${task}_32.json"
truth="$root/poc_llada2_flash_ep/data/bounded_eval_32/${task}_32_truth.json"
out="$root/poc_llada2_flash_ep/results/quality/${task}/${topology}/r${repeat}"
log="$root/poc_llada2_flash_ep/logs/quality_${task}_${topology}_r${repeat}.log"
mkdir -p "$out"
nvidia-smi -i 0,1,2,3 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

export CUDA_VISIBLE_DEVICES=0,1,2,3
common=(
  "$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py"
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3
  --batch_size 4 --gen_len 128 --block_length 32 --threshold 0.9
  --config 42 --model_type flash --mini_batch_size 4
  --output_dir "$out" --exp_name "quality_${task}_${topology}_r${repeat}"
)
if [[ "$topology" == "tp4" ]]; then
  "${common[@]}" --ep_size 1 --moe_a2a_backend none 2>&1 | tee "$log"
else
  export SGL_ENABLE_JIT_DEEPGEMM=false
  export SGLANG_ENABLE_JIT_DEEPGEMM=false
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
  "${common[@]}" --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal 2>&1 | tee "$log"
fi
prediction="$(find "$out" -maxdepth 1 -name '*.jsonl' -print -quit)"
"$venv/bin/python" "$root/poc_llada2_flash_ep/scripts/evaluate_bounded_quality.py" \
  --task "$task" --predictions "$prediction" --truth "$truth" --output "$out/quality.json"
