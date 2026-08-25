#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
result_dir="${repo_root}/poc_flashvep/deepep_revalidation/results/intra_expert_visual_redundancy_${run_id}"
capture_python="/home/esjung/.venvs/flashvep-deepep-v020/bin/python3"
analysis_python="/home/esjung/anaconda3/bin/python3"

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0
export FLASHVEP_FUNCTIONAL_LAYERS=4,12,24,36,44,47
export FLASHVEP_FUNCTIONAL_SAVE_HIDDEN=1
export LD_LIBRARY_PATH="/home/esjung/.cache/flashvep-deepep-v020/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${repo_root}/poc_flashvep/visual_expert_functional_redundancy/hooks:${repo_root}:${PYTHONPATH:-}"

"${capture_python}" -m poc_flashvep.intra_expert_visual_redundancy.capture --output-dir "${result_dir}"
PYTHONPATH="${repo_root}" "${analysis_python}" -m poc_flashvep.intra_expert_visual_redundancy.analyze --result-dir "${result_dir}"
echo "${result_dir}"
