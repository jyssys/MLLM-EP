#!/usr/bin/env bash
set -euo pipefail

task="$1"
sample_count="$2"
block_length="$3"
mini_batch_size="$4"
generation="$5"
threshold="$6"
regime="$7"
repeat="$8"
target_total_length="${9:-0}"

repo=/home/esjung/MLLM-EP-dynamic-block-deep
task_root="$repo/poc_ep_dynamic_block"
venv=/home/esjung/.venvs/llada2-flash-sglang-053
dinfer=/home/esjung/external/dinfer-llada2-dynamic-block
model=/home/esjung/models/LLaDA2.0-flash-744c3f8
if [[ "$sample_count" == "32" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval_32"
elif [[ "$sample_count" == "8" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/bounded_eval"
elif [[ "$sample_count" == "1" ]]; then
  data_dir="$repo/poc_llada2_flash_ep/data/smoke_eval"
else
  echo "unsupported sample count: $sample_count" >&2
  exit 2
fi
dataset="$data_dir/${task}_${sample_count}.json"
target_tag=""
target_args=()
if (( target_total_length > 0 )); then
  target_tag="_L${target_total_length}"
  target_args=(--target_total_length "$target_total_length")
fi
tag="${regime}_${task}_n${sample_count}_B${block_length}_m${mini_batch_size}_g${generation}_t${threshold}${target_tag}_r${repeat}"
out="$task_root/results/clean/$tag"
log="$task_root/logs/$tag.log"
gpu_log="$task_root/logs/${tag}_gpu.csv"
mkdir -p "$out" "$task_root/logs"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$dinfer/python:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
active_pids="$(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)"
if [[ -n "$active_pids" ]]; then
  echo "GPU 4-7 conflict; refusing launch" >&2
  ps -o user,pid,ppid,etime,args -p "$(echo "$active_pids" | paste -sd, -)" >&2 || true
  exit 3
fi

unset LLADA_DENOISE_TRACE LLADA_WAVE_TRACE LLADA_DISCOVERY_TRACE
unset LLADA_BLOCK_SCHEDULE
unset LLADA_TEMPORAL_COMM_TRACE_DIR LLADA_PP_BOUNDARY_TRACE_DIR
unset LLADA_EP_TRACE_DIR LLADA_EP_SHAPE_TRACE_DIR LLADA_BLOCK_TRACE_DIR
unset LLADA_LAYER_SENSITIVITY_TRACE_DIR LLADA_ASYNC_POLICY_TRACE_DIR
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_JIT_DEEPGEMM=false
local_source_rows=$((block_length * mini_batch_size / 4))
if (( block_length * mini_batch_size % 4 != 0 || local_source_rows > 1024 )); then
  echo "DeepEP normal supports at most 1024 source rows per EP rank; got local M=${local_source_rows}" >&2
  exit 4
fi
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

start_epoch=$(date +%s)
start_iso=$(date --iso-8601=seconds)
status=ok
set +e
"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size "$sample_count" --mini_batch_size "$mini_batch_size" \
  --gen_len "$generation" --block_length "$block_length" \
  "${target_args[@]}" \
  --threshold "$threshold" --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  >"$log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]] || ! rg -q '^Forward:' "$log"; then
  status="failed_rc${rc}"
fi
end_epoch=$(date +%s)
end_iso=$(date --iso-8601=seconds)
wall=$((end_epoch-start_epoch))
python - "$task_root/GPU_TIME_LOG.csv" "$start_iso" "$end_iso" "$wall" "$tag" "$status" <<'PY'
import csv, sys
path, start, end, wall, tag, status = sys.argv[1:]
with open(path, "a", newline="") as f:
    csv.writer(f, lineterminator="\n").writerow(
        [start, end, wall, 4, float(wall) * 4 / 3600, "static", tag, status]
    )
PY
python - "$task_root/ATTEMPT_LOG.csv" "$start_iso" "$task" "$sample_count" "$block_length" "$mini_batch_size" "$generation" "$threshold" "$regime" "$repeat" "$status" <<'PY'
import csv, sys
path, timestamp, task, n, block, mini, gen, threshold, regime, repeat, status = sys.argv[1:]
with open(path, "a", newline="") as f:
    csv.writer(f, lineterminator="\n").writerow(
        [timestamp, "static", task, n, block, mini, gen, threshold, regime, repeat, status, ""]
    )
PY
if [[ "$status" != "ok" ]]; then
  tail -120 "$log" >&2
  exit 1
fi
rg '^Forward:' "$log" | tail -1
