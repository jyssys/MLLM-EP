#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POC="$ROOT/poc_mllm_rank_aware_active_expert_reduction/q2_2_causal_visual_path"
PY=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTHONPATH="$ROOT"
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_DEEP_GEMM=0 VLLM_NO_USAGE_STATS=1

restore() {
  "$PY" -m poc_mllm_rank_aware_active_expert_reduction.gpu_guard start \
    --gpus 2,3,6,7 --record "$POC/artifacts/gpu_restore_reproduce.json" || true
}
trap restore EXIT INT TERM

"$PY" -m poc_mllm_rank_aware_active_expert_reduction.gpu_guard stop \
  --gpus 2,3,6,7 --record "$POC/artifacts/gpu_stop_reproduce.json"

# The complete command ledger, including append-only run names, is written to
# artifacts/reproducible_commands.json by the completed measured run. Raw runs
# are intentionally not overwritten by this convenience wrapper.
/home/esjung/anaconda3/bin/python -m pytest -q "$POC/tests"
