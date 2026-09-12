#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/esjung/MLLM-EP-vanilla-dllm-ep
PY=/home/esjung/.venvs/team-sdar-py311/bin/python
RUNNER="$ROOT/poc_team_positive_ep2/scripts/run_ep2_positive_control.py"
MODEL=/home/esjung/models/SDAR-30B-A3B-Chat-b32-c351bbc
TEAM=/home/esjung/external/TEAM-MoE-dLLM
PROMPTS="$ROOT/poc_team_positive_ep2/inputs/official_subset_4x4.jsonl"
OUT="$ROOT/poc_vanilla_dllm_ep/results/oracles/elastic_ep_microbench"

mkdir -p "$OUT/logs"
for ep in 4 2 1; do
  case "$ep" in
    1) visible=0 ;;
    2) visible=0,1 ;;
    4) visible=0,1,2,3 ;;
  esac
  stem="ep${ep}_layer24"
  CUDA_VISIBLE_DEVICES="$visible" "$PY" "$RUNNER" \
    --mode baseline \
    --model-dir "$MODEL" \
    --team-repo "$TEAM" \
    --prompts "$PROMPTS" \
    --output "$OUT/${stem}.json" \
    --world-size "$ep" \
    --local-expert-backend vllm_fused \
    --instrument \
    --microbenchmark-only \
    --microbenchmark-layer 24 \
    --microbenchmark-rows 1,2,4,8,16,32,64,128,256 \
    --microbenchmark-warmup 5 \
    --microbenchmark-repeats 30 \
    --seed 1234 \
    >"$OUT/logs/${stem}.log" 2>&1
done
