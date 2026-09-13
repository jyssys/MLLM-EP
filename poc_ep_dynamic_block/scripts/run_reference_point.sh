#!/usr/bin/env bash
set -euo pipefail

task="$1"
sample_count="$2"
block_length="$3"
generation="$4"
threshold="$5"
repeat="$6"
target_total_length="$7"

repo=/home/esjung/MLLM-EP-dynamic-block-deep
root="$repo/poc_ep_dynamic_block"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-dynamic-block
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
case "$sample_count" in
  32) data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval_32" ;;
  8) data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval" ;;
  1) data_dir="$repo/poc_llada2_flash_ep/data/smoke_eval" ;;
  *) echo "unsupported sample count: $sample_count" >&2; exit 2 ;;
esac
dataset="$data_dir/${task}_${sample_count}.json"
tag="reference_${task}_n${sample_count}_B${block_length}_m${sample_count}_g${generation}_t${threshold}_L${target_total_length}_r${repeat}"
out="$root/results/clean/$tag"
log="$root/logs/$tag.log"
gpu_log="$root/logs/${tag}_gpu.csv"
mkdir -p "$out" "$root/logs"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi
local_source_rows=$((block_length * sample_count / 4))
if (( block_length * sample_count % 4 != 0 || local_source_rows > 1024 )); then
  echo "DeepEP source-row cap exceeded: local M=${local_source_rows}" >&2
  exit 4
fi

export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
unset LLADA_BLOCK_SCHEDULE LLADA_DENOISE_TRACE LLADA_WAVE_TRACE LLADA_DISCOVERY_TRACE
unset LLADA_EP_TRACE_DIR LLADA_EP_SHAPE_TRACE_DIR LLADA_BLOCK_TRACE_DIR

nvidia-smi -i 4,5,6,7 \
  --query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 >"$gpu_log" &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

start_epoch=$(date +%s)
start_iso=$(date --iso-8601=seconds)
status=ok
set +e
"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size "$sample_count" --mini_batch_size "$sample_count" \
  --gen_len "$generation" --block_length "$block_length" \
  --target_total_length "$target_total_length" --threshold "$threshold" \
  --config 42 --model_type flash --output_dir "$out" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  --use_naive_batching >"$log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]] || ! rg -q '^Forward:' "$log"; then status="failed_rc${rc}"; fi
end_epoch=$(date +%s)
end_iso=$(date --iso-8601=seconds)
wall=$((end_epoch-start_epoch))
python - "$root/GPU_TIME_LOG.csv" "$start_iso" "$end_iso" "$wall" "$tag" "$status" <<'PY'
import csv, sys
path, start, end, wall, tag, status = sys.argv[1:]
with open(path, "a", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerow(
        [start, end, wall, 4, float(wall) * 4 / 3600, "reference", tag, status]
    )
PY
python - "$root/ATTEMPT_LOG.csv" "$start_iso" "$task" "$sample_count" "$block_length" "$sample_count" "$generation" "$threshold" "$repeat" "$status" <<'PY'
import csv, sys
path, timestamp, task, n, block, mini, generation, threshold, repeat, status = sys.argv[1:]
with open(path, "a", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerow(
        [timestamp, "reference", task, n, block, mini, generation, threshold, "naive", repeat, status, "correctness anchor"]
    )
PY
if [[ "$status" != ok ]]; then tail -120 "$log" >&2; exit 1; fi
rg '^Forward:' "$log" | tail -1
