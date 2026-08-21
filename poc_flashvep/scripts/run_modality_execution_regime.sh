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
capture=${CAPTURE:-/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt}
run_id=$(date +%Y%m%d_%H%M%S)
result_dir=${1:-${repo_root}/poc_flashvep/deepep_revalidation/results/modality_execution_regime_${run_id}}
report=${2:-${repo_root}/poc_flashvep/reports/flashvep_modality_execution_regime_report.md}

if [[ -e "${result_dir}" ]]; then
  echo "refusing to overwrite ${result_dir}" >&2
  exit 2
fi

export NVSHMEM_DIR="${deepep_workspace}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"

PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
"${deepep_venv}/bin/python" "${repo_root}/poc_flashvep/modality_execution_regime/prepare_and_capture.py" \
  prepare --model-path "${model}" --output-dir "${result_dir}" \
  2>&1 | tee "${result_dir}.prepare.log"
mv "${result_dir}.prepare.log" "${result_dir}/prepare.log"

FLASHVEP_VISION_TILE_CAPTURE_FIX=1 \
PYTHONPATH="${repo_root}/poc_flashvep/modality_execution_regime/capture_hooks:${repo_root}:${PYTHONPATH:-}" \
"${deepep_venv}/bin/python" \
  "${repo_root}/poc_flashvep/modality_execution_regime/prepare_and_capture.py" \
  capture --model-path "${model}" --output-dir "${result_dir}" \
  2>&1 | tee "${result_dir}/capture.log"

mkdir -p "${result_dir}/replay" "${result_dir}/backend_proof"
export FLASHVEP_MODALITY_RESULT_DIR="${result_dir}"
export FLASHVEP_MODALITY_REPLAY_DIR="${result_dir}/replay"
export FLASHVEP_MODALITY_CAPTURE_PATH="${capture}"
export FLASHVEP_MODALITY_WARMUPS=${MODALITY_WARMUPS:-3}
export FLASHVEP_MODALITY_ITERATIONS=${MODALITY_ITERATIONS:-15}
export FLASHVEP_DEEPEP_PROOF_DIR="${result_dir}/backend_proof"
export FLASHVEP_CONFIGURED_ALL2ALL_BACKEND=deepep_high_throughput
export FLASHVEP_CONFIGURED_DBO=false

PYTHONPATH="${repo_root}/poc_flashvep/modality_execution_regime/hooks:${repo_root}:${PYTHONPATH:-}" \
"${deepep_venv}/bin/python" \
  "${repo_root}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
  --model-path "${model}" --output "${result_dir}/replay_trigger.json" \
  --all2all-backend deepep_high_throughput \
  --request-counts 1 --warmups 0 --iterations 1 \
  2>&1 | tee "${result_dir}/replay.log"

PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
python3 "${repo_root}/poc_flashvep/modality_execution_regime/analyze.py" \
  "${result_dir}" --report "${report}" \
  2>&1 | tee "${result_dir}/analysis.log"

printf '%s\n' "${result_dir}"
