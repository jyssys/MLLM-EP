#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/esjung/MLLM-EP-vanilla-dllm-ep
PY=/home/esjung/.venvs/team-sdar-py311/bin/python
OUT="$ROOT/poc_vanilla_dllm_ep/results/stage0_clean"
mapfile -t inputs < <(find "$OUT" -maxdepth 1 -name 'vanilla_*_rank0.json' | sort)
"$PY" "$ROOT/poc_team_positive_ep2/scripts/score_positive_control.py" \
  "${inputs[@]}" \
  --output "$OUT/quality_scores.json" \
  --dataset "$ROOT/poc_team_positive_ep2/inputs/official_subset_4x4.jsonl" \
  --human-eval-dir "/home/esjung/external/TEAM-MoE-dLLM/evaluation/opencompass/human-eval"
