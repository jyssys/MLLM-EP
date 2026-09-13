#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?dataset (gsm8k|humaneval)}"
waves="${2:?comma-separated refinement waves}"
repeat="${3:?restart id}"
result_root="${4:-/home/esjung/MLLM-EP-exact-overlap/poc_dllm_exact_overlap/results/exact_overlap_20260913_134622}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/esjung/external/dinfer-llada2-exact-overlap/python:${PYTHONPATH:-}
export LLADA_EXACT_OVERLAP_PHASE_WAVES="$waves"
export LLADA_EXACT_OVERLAP_LAYER=16
export LLADA_EXACT_OVERLAP_WARMUPS=5
export LLADA_EXACT_OVERLAP_REPS=30
export LLADA_DENOISE_TRACE=1
export LLADA_WAVE_TRACE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

if nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader | rg -q '[0-9]'; then
  echo "Refusing launch: a compute process already owns one or more visible GPUs." >&2
  exit 2
fi

case "$dataset" in
  gsm8k)
    input=/home/esjung/MLLM-EP-exact-overlap/poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32.json
    ;;
  humaneval)
    input=/home/esjung/MLLM-EP-exact-overlap/poc_llada2_flash_ep/data/bounded_eval_32/humaneval_32.json
    ;;
  *)
    echo "Unknown dataset: $dataset" >&2
    exit 2
    ;;
esac

wave_tag="${waves//,/-}"
probe_dir="$result_root/phase_pair/$dataset/waves${wave_tag}/r${repeat}"
log_dir="$result_root/logs"
mkdir -p "$probe_dir" "$log_dir"
export LLADA_EXACT_OVERLAP_DIR="$probe_dir"

log="$log_dir/phase_pair_${dataset}_waves${wave_tag}_layer16_r${repeat}.log"
gpu_log="$log_dir/phase_pair_${dataset}_waves${wave_tag}_layer16_r${repeat}_gpu.csv"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used \
      --format=csv,noheader,nounits
    sleep 0.2
  done
) >"$gpu_log" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

/home/esjung/.venvs/llada2-flash-sglang-053/bin/python \
  /home/esjung/external/dinfer-llada2-exact-overlap/benchmarks/benchmark_dataset_sglang.py \
  --model_name /home/esjung/models/LLaDA2.0-flash-744c3f8 \
  --dataset "$input" \
  --gpu 0,1,2,3 \
  --batch_size 32 \
  --mini_batch_size 32 \
  --gen_len 32 \
  --block_length 32 \
  --threshold 0.9 \
  --config 42 \
  --model_type flash \
  --output_dir "$probe_dir" \
  --exp_name "phase_pair_${dataset}_ep4_b32_mini32_r${repeat}_g32" \
  --ep_size 4 \
  --moe_a2a_backend deepep \
  --deepep_mode normal 2>&1 | tee "$log"

rg -q '^Forward:' "$log"
test -f "$probe_dir/phase_pair_rank0.json"
