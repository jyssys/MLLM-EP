#!/usr/bin/env bash
set -euo pipefail

result_root="$1"
topology="${2:-pp2_ep2}"
repo=/home/esjung/MLLM-EP-temporal-comm-async-pp
dinfer=/home/esjung/external/dinfer-llada2-temporal-comm-async
venv=/home/esjung/.venvs/llada2-flash-sglang-053
case "$topology" in
  pp2_ep2) ep=2; pp=2; backend=deepep ;;
  pp4) ep=1; pp=4; backend=none ;;
  *) echo "unknown topology: $topology" >&2; exit 2 ;;
esac
out="$result_root/trackB/${topology}_load_audit"
log="$result_root/logs/trackB_${topology}_load_audit.log"
mkdir -p "$out" "$result_root/logs"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name /home/esjung/models/LLaDA2.0-flash-744c3f8 \
  --dataset "$repo/poc_llada2_flash_ep/data/bounded_eval_32/gsm8k_32.json" \
  --gpu 0,1,2,3 --batch_size 32 --mini_batch_size 32 \
  --gen_len 32 --block_length 32 --threshold 0.9 --config 42 \
  --model_type flash --output_dir "$out" --exp_name "${topology}_load_audit" \
  --ep_size "$ep" --pp_size "$pp" --moe_a2a_backend "$backend" --deepep_mode normal \
  --load_only >"$log" 2>&1

test "$(find "$out" -name 'load_audit_rank*.json' | wc -l)" -eq 4
