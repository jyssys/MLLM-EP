#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/esjung/MLLM-EP-vanilla-dllm-ep
PY=/home/esjung/.venvs/team-sdar-py311/bin/python
RUNNER="$ROOT/poc_team_positive_ep2/scripts/run_ep2_positive_control.py"
MODEL=/home/esjung/models/SDAR-30B-A3B-Chat-b32-c351bbc
TEAM=/home/esjung/external/TEAM-MoE-dLLM
PROMPTS="$ROOT/poc_team_positive_ep2/inputs/official_subset_4x4.jsonl"
OUT="$ROOT/poc_vanilla_dllm_ep/results/stage1_timing"

mkdir -p "$OUT/logs"
run_one() {
  local ep="$1"
  local visible
  case "$ep" in
    1) visible=0 ;;
    2) visible=0,1 ;;
    4) visible=0,1,2,3 ;;
    *) return 2 ;;
  esac
  local stem="vanilla_ep${ep}_timing"
  CUDA_VISIBLE_DEVICES="$visible" "$PY" "$RUNNER" \
    --mode baseline \
    --model-dir "$MODEL" \
    --team-repo "$TEAM" \
    --prompts "$PROMPTS" \
    --output "$OUT/${stem}.json" \
    --ids 'gsm8k_0,HumanEval/0' \
    --world-size "$ep" \
    --local-expert-backend vllm_fused \
    --gen-length 128 \
    --block-length 32 \
    --denoising-steps 32 \
    --threshold 1.0 \
    --disable-early-stop \
    --warmup 1 \
    --seed 1234 \
    --instrument \
    >"$OUT/logs/${stem}.log" 2>&1
}

# Separate low-observer-tax timing: no route/output tensor capture.
run_one 2
run_one 4
run_one 1
