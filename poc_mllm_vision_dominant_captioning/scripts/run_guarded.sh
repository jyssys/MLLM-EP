#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 4,5[,6,7] command..." >&2
  exit 2
fi
GPU_SET="$1"
shift
ROOT="/home/esjung/MLLM-EP-long-reasoning-signal"
PY="/home/esjung/.venvs/flashvep-deepep-v020/bin/python"
STAMP="$(date +%s)"
restore() {
  PYTHONPATH="$ROOT" "$PY" -m poc_mllm_vision_dominant_captioning.gpu_guard start \
    --gpus "$GPU_SET" --record "$ROOT/poc_mllm_vision_dominant_captioning/artifacts/gpu_restart_${STAMP}.json" || true
}
trap restore EXIT INT TERM
PYTHONPATH="$ROOT" "$PY" -m poc_mllm_vision_dominant_captioning.gpu_guard stop \
  --gpus "$GPU_SET" --record "$ROOT/poc_mllm_vision_dominant_captioning/artifacts/gpu_stop_${STAMP}.json"
"$@"
