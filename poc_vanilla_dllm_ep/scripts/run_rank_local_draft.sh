#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/esjung/MLLM-EP-vanilla-dllm-ep
PY=/home/esjung/.venvs/team-sdar-py311/bin/python
RUNNER="$ROOT/poc_team_positive_ep2/scripts/run_ep2_positive_control.py"
MODEL=/home/esjung/models/SDAR-30B-A3B-Chat-b32-c351bbc
TEAM=/home/esjung/external/TEAM-MoE-dLLM
PROMPTS="$ROOT/poc_team_positive_ep2/inputs/official_subset_4x4.jsonl"
OUT="$ROOT/poc_vanilla_dllm_ep/results/oracles/rank_local_draft"

mkdir -p "$OUT/logs"
for draft_k in 1 2 4; do
  stem="ep4_local_draft_k${draft_k}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" "$RUNNER" \
    --mode baseline \
    --model-dir "$MODEL" \
    --team-repo "$TEAM" \
    --prompts "$PROMPTS" \
    --output "$OUT/${stem}.json" \
    --ids gsm8k_0 \
    --world-size 4 \
    --local-expert-backend vllm_fused \
    --gen-length 64 \
    --block-length 32 \
    --denoising-steps 32 \
    --threshold 1.0 \
    --disable-early-stop \
    --warmup 1 \
    --seed 1234 \
    --capture-local-draft \
    --local-draft-k "$draft_k" \
    >"$OUT/logs/${stem}.log" 2>&1
done
