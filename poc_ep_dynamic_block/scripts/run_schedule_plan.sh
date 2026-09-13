#!/usr/bin/env bash
set -euo pipefail

task="$1"
sample_count="$2"
mini_batch_size="$3"
generation="$4"
threshold="$5"
target_total_length="$6"
plan="$7"
repeat="$8"
execution_batch_size="${9:-$sample_count}"
barrier_mode="${10:-0}"

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
plan_name="$(basename "$plan" .json)"
barrier_tag=""
if [[ "$barrier_mode" == "1" ]]; then barrier_tag="_barrier1"; fi
tag="schedule_${plan_name}_${task}_n${sample_count}_m${mini_batch_size}_L${target_total_length}_t${threshold}_r${repeat}${barrier_tag}"
out="$root/results/schedules/$tag"
log="$root/logs/$tag.log"
gpu_log="$root/logs/${tag}_gpu.csv"
mkdir -p "$out" "$root/logs"

max_block="$($venv/bin/python - "$plan" <<'PY'
import json, sys
policies=json.load(open(sys.argv[1]))
values=[]
for policy in policies:
    raw=policy.get('block_schedule', [])
    if isinstance(raw, str): raw=[int(x) for x in raw.split(',') if x.strip()]
    values.extend(int(x) for x in raw)
print(max(values or [32]))
PY
)"
local_source_rows=$((max_block * mini_batch_size / 4))
if (( max_block * mini_batch_size % 4 != 0 || local_source_rows > 1024 )); then
  echo "schedule violates DeepEP source-row cap: local M=${local_source_rows}" >&2
  exit 4
fi

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
unset LLADA_EP_TRACE_DIR LLADA_EP_SHAPE_TRACE_DIR LLADA_BLOCK_TRACE_DIR
unset LLADA_BLOCK_SCHEDULE
export LLADA_BLOCK_SCHEDULE_BARRIER="$barrier_mode"
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

start_epoch=$(date +%s)
start_iso=$(date --iso-8601=seconds)
status=ok
set +e
"$venv/bin/python" "$dinfer/benchmarks/benchmark_dataset_sglang.py" \
  --model_name "$model" --dataset "$dataset" --gpu 0,1,2,3 \
  --batch_size "$execution_batch_size" --mini_batch_size "$mini_batch_size" \
  --gen_len "$generation" --block_length 32 \
  --target_total_length "$target_total_length" \
  --threshold "$threshold" --config 42 --model_type flash \
  --output_dir "$out" --exp_name "$tag" \
  --intervention_plan "$plan" \
  --ep_size 4 --moe_a2a_backend deepep --deepep_mode normal \
  >"$log" 2>&1
rc=$?
set -e
expected="$($venv/bin/python - "$plan" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)"
actual=$(find "$out" -maxdepth 1 -name 'policy_*.json' | wc -l)
if [[ "$rc" -ne 0 || "$actual" -ne "$expected" ]]; then status="failed_rc${rc}_policies${actual}of${expected}"; fi
end_epoch=$(date +%s)
end_iso=$(date --iso-8601=seconds)
wall=$((end_epoch-start_epoch))
python - "$root/GPU_TIME_LOG.csv" "$start_iso" "$end_iso" "$wall" "$tag" "$status" <<'PY'
import csv, sys
path, start, end, wall, tag, status = sys.argv[1:]
with open(path, "a", newline="") as f:
    csv.writer(f, lineterminator="\n").writerow(
        [start, end, wall, 4, float(wall)*4/3600, "schedule", tag, status]
    )
PY
python - "$root/ATTEMPT_LOG.csv" "$start_iso" "$task" "$sample_count" "$generation" "$threshold" "$plan_name" "$repeat" "$status" <<'PY'
import csv, sys
path, ts, task, n, gen, threshold, plan, repeat, status = sys.argv[1:]
with open(path, "a", newline="") as f:
    csv.writer(f, lineterminator="\n").writerow(
        [ts, "schedule", task, n, "mixed", "plan", gen, threshold, plan, repeat, status, "actual trajectories; barrier=" + __import__("os").environ.get("LLADA_BLOCK_SCHEDULE_BARRIER", "0")]
    )
PY
if [[ "$status" != ok ]]; then tail -160 "$log" >&2; exit 1; fi
find "$out" -maxdepth 1 -name 'policy_*.json' -printf '%f\n' | sort
