#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4,5,6,7" ]]; then
  echo "GPU safety violation" >&2
  exit 2
fi

result_root=${1:?usage: run_route_replay_matrix.sh RESULT_ROOT}
repo=/home/esjung/MLLM-EP-rankfanout
python=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
model=/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
route_run="$result_root/real/profile_r1_agrs"
mkdir -p "$result_root/replay/logs"

for restart in ${RANKFANOUT_RESTARTS:-1 2 3}; do
  if [[ "$restart" == 2 ]]; then backends=(deepep_ht agrs); else backends=(agrs deepep_ht); fi
  for backend in "${backends[@]}"; do
    output="$result_root/replay/restart${restart}_${backend}/results.jsonl"
    log="$result_root/replay/logs/restart${restart}_${backend}.log"
    echo "START replay-r${restart}-${backend} $(date --iso-8601=seconds)"
    "$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$repo/poc_rankfanout/scripts/bench_real_route_replay.py" \
      --model "$model" --route-run "$route_run" --route-run-id profile-r1-agrs \
      --iteration 1 --output "$output" --backend "$backend" \
      --warmup 3 --reps 10 >"$log" 2>&1
    echo "DONE replay-r${restart}-${backend} $(date --iso-8601=seconds)"
  done
done
