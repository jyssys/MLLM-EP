#!/usr/bin/env bash
set -euo pipefail

result_root="$1"
repo=/home/esjung/MLLM-EP-temporal-comm-async-pp
venv=/home/esjung/.venvs/llada2-flash-sglang-053
out="$result_root/trackA/deepep_payload_sweep"
mkdir -p "$out"

export CUDA_VISIBLE_DEVICES=4,5,6,7
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi

for source_tokens in 1 2 4 8 16 32 64 128 256 512 1024; do
  "$venv/bin/python" "$repo/poc_llada2_flash_ep/scripts/deepep_ep4_smoke.py" \
    --output "$out/source_tokens_${source_tokens}" --tokens "$source_tokens" \
    --hidden 4096 --experts 256 --topk 8 --warmup 8 --repeats 30
done
"$venv/bin/python" "$repo/poc_llada2_flash_ep/scripts/summarize_deepep_sweep.py" \
  "$out" "$out/summary.csv"
