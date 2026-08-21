#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/esjung/MLLM-EP-github"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
result_dir="$repo_dir/poc_flashvep/deepep_revalidation/results/prerouter_visual_signal_${run_id}"
env_dir="${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}"
deepep_workspace="${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export PYTHONPATH="$repo_dir/poc_flashvep/prerouter_visual_signal/hooks:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR="$deepep_workspace/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"

cd "$repo_dir"
"$env_dir/bin/python" -m poc_flashvep.prerouter_visual_signal.run_capture --output-dir "$result_dir"
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" python3 -m poc_flashvep.prerouter_visual_signal.analyze --result-dir "$result_dir"
