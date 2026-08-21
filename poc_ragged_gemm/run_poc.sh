#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "4" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=4 for the single-GPU causal benchmark." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
result_dir="${repo_dir}/poc_ragged_gemm/results/sonic_ep_poc_${run_id}"
python_bin="${PYTHON_BIN:-/home/esjung/.venvs/flashvep-deepep-v020/bin/python}"
mkdir -p "${result_dir}"

"${python_bin}" "${repo_dir}/poc_ragged_gemm/benchmark_sonic.py" \
  --output "${result_dir}/synthetic.json"
"${python_bin}" "${repo_dir}/poc_ragged_gemm/analyze.py" \
  --result-dir "${result_dir}"

echo "${result_dir}"
