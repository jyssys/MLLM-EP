#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
result_dir="${repo_root}/poc_flashvep/deepep_revalidation/results/causal_modality_wavefront_${run_id}"
report="${repo_root}/poc_flashvep/reports/causal_modality_wavefront.md"
source_dir="${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
model="/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
capture="/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt"
venv="/home/esjung/.venvs/flashvep-deepep-v020"
workspace="/home/esjung/.cache/flashvep-deepep-v020"

if [[ -e "${result_dir}" ]]; then
  echo "refusing to overwrite ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}/timing" "${result_dir}/backend_proof"

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR="${workspace}/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"
export FLASHVEP_CAUSAL_WAVEFRONT_ENABLE=1
export FLASHVEP_CAUSAL_SOURCE_DIR="${source_dir}"
export FLASHVEP_CAUSAL_CAPTURE_PATH="${capture}"
export FLASHVEP_CAUSAL_OUTPUT_DIR="${result_dir}/timing"
export FLASHVEP_CAUSAL_MODE=timing
export FLASHVEP_CAUSAL_WARMUPS="${CAUSAL_WARMUPS:-2}"
export FLASHVEP_CAUSAL_ITERATIONS="${CAUSAL_ITERATIONS:-7}"
export FLASHVEP_CAUSAL_MAX_REQUESTS="${CAUSAL_MAX_REQUESTS:-24}"
export FLASHVEP_DEEPEP_PROOF_DIR="${result_dir}/backend_proof"
export FLASHVEP_CONFIGURED_ALL2ALL_BACKEND=deepep_high_throughput
export FLASHVEP_CONFIGURED_DBO=false

PYTHONPATH="${repo_root}/poc_flashvep/causal_modality_wavefront/hooks:${repo_root}:${PYTHONPATH:-}" \
  "${venv}/bin/python" "${repo_root}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
  --model-path "${model}" --output "${result_dir}/trigger.json" \
  --all2all-backend deepep_high_throughput --request-counts 1 \
  --warmups 0 --iterations 1 2>&1 | tee "${result_dir}/timing.log"

PYTHONPATH="${repo_root}" /home/esjung/anaconda3/bin/python3 \
  -m poc_flashvep.causal_modality_wavefront.analyze \
  --result-dir "${result_dir}" --report "${report}" \
  2>&1 | tee "${result_dir}/analysis.log"

status=$(python3 -c "import json; print(json.load(open('${result_dir}/summary.json'))['CAUSAL_MODALITY_WAVEFRONT'])")
if [[ "${status}" == "GO" || "${status}" == "HOLD" ]]; then
  mkdir -p "${result_dir}/diagnostic"
  export FLASHVEP_CAUSAL_OUTPUT_DIR="${result_dir}/diagnostic"
  export FLASHVEP_CAUSAL_MODE=diagnostic
  PYTHONPATH="${repo_root}/poc_flashvep/causal_modality_wavefront/hooks:${repo_root}:${PYTHONPATH:-}" \
    "${venv}/bin/python" "${repo_root}/poc_flashvep/deepep_revalidation/vllm_backend_matrix.py" \
    --model-path "${model}" --output "${result_dir}/diagnostic_trigger.json" \
    --all2all-backend deepep_high_throughput --request-counts 1 \
    --warmups 0 --iterations 1 2>&1 | tee "${result_dir}/diagnostic.log"
  PYTHONPATH="${repo_root}" /home/esjung/anaconda3/bin/python3 \
    -m poc_flashvep.causal_modality_wavefront.analyze \
    --result-dir "${result_dir}" --report "${report}" \
    2>&1 | tee "${result_dir}/analysis_with_diagnostic.log"
fi

echo "${result_dir}"
