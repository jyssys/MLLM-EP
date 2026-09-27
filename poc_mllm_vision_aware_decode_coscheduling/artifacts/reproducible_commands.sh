#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/esjung/MLLM-EP-long-reasoning-signal}"
cd "${REPO_ROOT}"
export PYTHONPATH=.

python -m poc_mllm_vision_aware_decode_coscheduling.scripts.run_stage_a \
  --cohort poc_mllm_vision_aware_decode_coscheduling/artifacts/frozen_request_pool.json \
  --compact-trace poc_mllm_vision_aware_decode_coscheduling/artifacts/source_pool_trace.jsonl.gz \
  --prior-analysis poc_mllm_vision_aware_decode_coscheduling/artifacts/predictor_contract.json \
  --expert-contract poc_mllm_vision_aware_decode_coscheduling/artifacts/expert_tensor_contract.json \
  --output poc_mllm_vision_aware_decode_coscheduling/artifacts

python -m poc_mllm_vision_aware_decode_coscheduling.scripts.render_reports \
  --root poc_mllm_vision_aware_decode_coscheduling \
  --baseline-audit poc_mllm_vision_aware_decode_coscheduling/artifacts/baseline_audit_reused.json \
  --source-spec /home/esjung/MLLM-EP-github/poc_flashvep/reports/MLLM_Vision_Aware_Decode_CoScheduling_TRUE_EP4_PoC.md

python -m poc_mllm_vision_aware_decode_coscheduling.scripts.verify_artifacts \
  --root poc_mllm_vision_aware_decode_coscheduling
