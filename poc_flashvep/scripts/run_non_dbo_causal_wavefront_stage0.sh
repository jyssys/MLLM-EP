#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
result="${repo_root}/poc_flashvep/deepep_revalidation/results/non_dbo_causal_wavefront_${run_id}"
previous="${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
model="/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
venv="/home/esjung/.venvs/flashvep-deepep-v020"
hook="${repo_root}/poc_flashvep/non_dbo_causal_wavefront/hooks"
report="${repo_root}/poc_flashvep/reports/non_dbo_causal_wavefront.md"

[[ "$(git -C "${repo_root}" branch --show-current)" == "flashvep/non-dbo-causal-wavefront" ]]
[[ -z "$(git -C "${repo_root}" status --short --untracked-files=no)" ]]
[[ ! -e "${result}" ]]
code_sha="$(git -C "${repo_root}" rev-parse HEAD)"
mkdir -p "${result}"

export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_DIR="/home/esjung/.cache/flashvep-deepep-v020/nvshmem"
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}"

for variant in A S; do
  PYTHONPATH="${hook}:${repo_root}:${PYTHONPATH:-}" "${venv}/bin/python" \
    -m poc_flashvep.non_dbo_causal_wavefront.run_stage0 \
    --variant "${variant}" --previous "${previous}" \
    --output-dir "${result}/${variant}" --model-path "${model}" \
    --code-sha "${code_sha}" 2>&1 | tee "${result}/${variant}.log"
done

PYTHONPATH="${repo_root}" /home/esjung/anaconda3/bin/python3 \
  -m poc_flashvep.non_dbo_causal_wavefront.analyze_stage0 \
  --result-dir "${result}" --report "${report}" \
  2>&1 | tee "${result}/analysis.log"

echo "${result}"
