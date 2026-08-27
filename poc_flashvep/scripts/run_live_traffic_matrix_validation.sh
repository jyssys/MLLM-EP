#!/usr/bin/env bash
set -euo pipefail

# Physical GPUs 1,2,3,4 are logical devices 0,1,2,3 in every worker.
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
venv=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
workspace=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
model=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
previous=${PREVIOUS_RESULT:-${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
result_dir=${1:-${repo_root}/poc_flashvep/deepep_revalidation/results/live_traffic_matrix_${run_id}}
mode=${2:-instrumented}

if [[ -e "${result_dir}" ]]; then echo "refusing to overwrite ${result_dir}" >&2; exit 2; fi
export NVSHMEM_DIR="${workspace}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export FLASHVEP_CONFIGURED_ALL2ALL_BACKEND=deepep_high_throughput
export FLASHVEP_CONFIGURED_DBO=false

extra=()
if [[ "${mode}" == "instrumented" ]]; then extra+=(--instrument); fi
PYTHONPATH="${repo_root}/poc_flashvep/live_traffic_matrix_validation/hooks:${repo_root}:${PYTHONPATH:-}" \
"${venv}/bin/python" "${repo_root}/poc_flashvep/live_traffic_matrix_validation/run_live.py" \
  --previous "${previous}" --output-dir "${result_dir}" --model-path "${model}" \
  --warmups "${WARMUPS:-2}" --iterations "${ITERATIONS:-2}" "${extra[@]}" 2>&1 | tee "${result_dir}.log"
mv "${result_dir}.log" "${result_dir}/live.log"
printf '%s\n' "${result_dir}"
