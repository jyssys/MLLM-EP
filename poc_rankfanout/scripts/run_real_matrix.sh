#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4,5,6,7" ]]; then
  echo "GPU safety violation: expected CUDA_VISIBLE_DEVICES=4,5,6,7" >&2
  exit 2
fi

mode=${1:?usage: run_real_matrix.sh clean|profile RESULT_ROOT}
result_root=${2:?usage: run_real_matrix.sh clean|profile RESULT_ROOT}
repo=/home/esjung/MLLM-EP-rankfanout
python=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
model=/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
cases='[{"name":"text512_c2","kind":"text","length":512,"concurrency":2,"output":1},{"name":"text2048_c2","kind":"text","length":2048,"concurrency":2,"output":1},{"name":"text8192_c2","kind":"text","length":8192,"concurrency":2,"output":1},{"name":"image448_c2","kind":"image","image_size":448,"concurrency":2,"output":1},{"name":"image1024_c2","kind":"vision_heavy","image_size":1024,"concurrency":2,"output":1}]'

if [[ "$mode" == clean ]]; then
  repetitions=3
  profile_args=()
elif [[ "$mode" == profile ]]; then
  repetitions=2
  profile_args=(--capture-routes --profile-moe)
else
  echo "unknown mode: $mode" >&2
  exit 2
fi

mkdir -p "$result_root/real/logs"
for restart in ${RANKFANOUT_RESTARTS:-1 2 3}; do
  if [[ "$restart" == 2 ]]; then
    backends=(deepep_high_throughput allgather_reducescatter)
  else
    backends=(allgather_reducescatter deepep_high_throughput)
  fi
  for backend in "${backends[@]}"; do
    if [[ "$backend" == allgather_reducescatter ]]; then short=agrs; else short=deepep; fi
    run_id="${mode}-r${restart}-${short}"
    output="$result_root/real/${mode}_r${restart}_${short}"
    log="$result_root/real/logs/${run_id}.log"
    echo "START $run_id $(date --iso-8601=seconds)"
    PYTHONPATH="$repo" "$python" "$repo/poc_rankfanout/scripts/bench_real_static_backends.py" \
      --model "$model" --output "$output" --backend "$backend" \
      --run-id "$run_id" --cases "$cases" --warmup 1 \
      --repetitions "$repetitions" --max-model-len 16384 \
      --max-num-batched-tokens 16384 --max-num-seqs 8 --kv-cache-gb 20 \
      "${profile_args[@]}" >"$log" 2>&1
    echo "DONE $run_id $(date --iso-8601=seconds)"
  done
done
