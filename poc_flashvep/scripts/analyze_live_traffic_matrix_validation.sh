#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result=${1:?result directory}
baseline=${2:-}
args=("${result}" --previous "${PREVIOUS_RESULT:-${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609}" --report "${repo_root}/poc_flashvep/reports/deepep_traffic_matrix_live.md")
if [[ -n "${baseline}" ]]; then args+=(--baseline "${baseline}"); fi
PYTHONPATH="${repo_root}:${PYTHONPATH:-}" python3 "${repo_root}/poc_flashvep/live_traffic_matrix_validation/analyze.py" "${args[@]}"
