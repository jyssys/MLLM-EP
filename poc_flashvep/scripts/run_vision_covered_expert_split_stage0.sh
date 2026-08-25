#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="${1:-$(date +%Y%m%d_%H%M%S)}"
source_dir="${repo_root}/poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"
output_dir="${repo_root}/poc_flashvep/deepep_revalidation/results/vision_covered_expert_split_${run_id}"

# Stage 0 is CPU-only analysis of an existing validated trace. If it fails,
# the preregistered protocol forbids launching Stage 1-3 GPU benchmarks.
PYTHONPATH="${repo_root}" /home/esjung/anaconda3/bin/python3 \
  -m poc_flashvep.vision_covered_expert_split.analyze_stage0 \
  --source "${source_dir}" --output-dir "${output_dir}"

echo "${output_dir}"
