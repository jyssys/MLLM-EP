#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
deepep_venv=${DEEPEP_VENV:-/home/esjung/.venvs/flashvep-deepep-v020}
deepep_workspace=${DEEPEP_WORKSPACE:-/home/esjung/.cache/flashvep-deepep-v020}
model=${MODEL:-/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c}
previous=${PREVIOUS_RESULT:-${repo_root}/poc_flashvep/deepep_revalidation/results/modality_execution_regime_20260821_102147}
run_id=$(date +%Y%m%d_%H%M%S)
result_dir=${1:-${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_${run_id}}
report=${2:-${repo_root}/poc_flashvep/reports/flashvep_live_prefill_execution_regime_report.md}

if [[ -e "${result_dir}" ]]; then
  echo "refusing to overwrite ${result_dir}" >&2
  exit 2
fi

export NVSHMEM_DIR="${deepep_workspace}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export FLASHVEP_CONFIGURED_ALL2ALL_BACKEND=deepep_high_throughput
export FLASHVEP_CONFIGURED_DBO=false

PYTHONPATH="${repo_root}/poc_flashvep/live_prefill_execution_regime/hooks:${repo_root}:${PYTHONPATH:-}" \
"${deepep_venv}/bin/python" "${repo_root}/poc_flashvep/live_prefill_execution_regime/run_live.py" \
  --previous "${previous}" --output-dir "${result_dir}" --model-path "${model}" \
  --warmups 3 --iterations 15 2>&1 | tee "${result_dir}.live.log"
mv "${result_dir}.live.log" "${result_dir}/live.log"

PYTHONPATH="${repo_root}:${PYTHONPATH:-}" python3 \
  "${repo_root}/poc_flashvep/live_prefill_execution_regime/analyze_live.py" \
  "${result_dir}" --previous "${previous}" --report "${report}" \
  2>&1 | tee "${result_dir}/analysis.log"

gzip -9 "${result_dir}"/raw_live/rank*.jsonl

printf '%s\n' "${result_dir}"
