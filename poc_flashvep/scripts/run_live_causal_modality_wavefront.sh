#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
result="${repo_root}/poc_flashvep/deepep_revalidation/results/live_causal_modality_wavefront_${run_id}"
previous="${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
model="/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
venv="/home/esjung/.venvs/flashvep-deepep-v020"
hook="${repo_root}/poc_flashvep/live_causal_modality_wavefront/hooks"
report="${repo_root}/poc_flashvep/reports/live_causal_modality_wavefront.md"

if [[ -e "${result}" ]]; then
  echo "refusing to overwrite ${result}" >&2
  exit 2
fi
mkdir -p "${result}"

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR="/home/esjung/.cache/flashvep-deepep-v020/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"

for mode in stock wavefront; do
  PYTHONPATH="${hook}:${repo_root}:${PYTHONPATH:-}" "${venv}/bin/python" \
    -m poc_flashvep.live_causal_modality_wavefront.run_mode \
    --mode "${mode}" --previous "${previous}" \
    --output-dir "${result}/${mode}" --model-path "${model}" \
    --warmups "${LIVE_WAVEFRONT_WARMUPS:-2}" \
    --iterations "${LIVE_WAVEFRONT_ITERATIONS:-7}" \
    --max-requests "${LIVE_WAVEFRONT_MAX_REQUESTS:-24}" \
    2>&1 | tee "${result}/${mode}.log"
done

PYTHONPATH="${repo_root}" /home/esjung/anaconda3/bin/python3 \
  -m poc_flashvep.live_causal_modality_wavefront.analyze \
  --result-dir "${result}" --report "${report}" \
  2>&1 | tee "${result}/analysis.log"

echo "${result}"
