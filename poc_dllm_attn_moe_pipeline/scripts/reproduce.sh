#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=2,3,6,7
cpu_python=/home/esjung/anaconda3/bin/python
gpu_python=/home/esjung/.venvs/llada2-flash-sglang-053/bin/python
# Analysis-only default does not acquire GPU resources or overwrite raw captures.
"$cpu_python" -m poc_dllm_attn_moe_pipeline.scripts.pack_evidence --restore
"$cpu_python" -m poc_dllm_attn_moe_pipeline.scripts.estimate_schedule_headroom
"$cpu_python" -m poc_dllm_attn_moe_pipeline.scripts.report
"$cpu_python" -m poc_dllm_attn_moe_pipeline.scripts.audit_evidence
"$gpu_python" -m pytest -q poc_dllm_attn_moe_pipeline/tests
# Fresh physical acquisition uses a NEW output dir. Never overwrite raw/main_v1.
# "$gpu_python" -m poc_dllm_attn_moe_pipeline.scripts.collect_readiness --output reproduce_new --limit 12
