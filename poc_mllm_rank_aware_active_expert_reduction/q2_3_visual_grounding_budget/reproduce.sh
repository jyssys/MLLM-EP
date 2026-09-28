#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
VENV=/home/esjung/.venvs/flashvep-deepep-v020/bin/python
export PYTHONPATH="$REPO"

CAPTION="$REPO/poc_mllm_vision_dominant_captioning"
Q22="$REPO/poc_mllm_rank_aware_active_expert_reduction/q2_2_causal_visual_path"
ART="$ROOT/artifacts"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.audit_baseline \
  --prior-rank-aware "$REPO/poc_mllm_rank_aware_active_expert_reduction/artifacts/final_verdict.json" \
  --prior-q22 "$Q22/artifacts/final_verdict.json" --output "$ART/baseline_audit.json"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.build_cohort \
  --image-cohort "$CAPTION/artifacts/manifests/frozen_image_cohort.json" \
  --natural "$CAPTION/artifacts/ep4_natural_metadata.jsonl.gz" \
  --prior-q2 "$REPO/poc_mllm_rank_aware_active_expert_reduction/artifacts/q2_vds_damage_joined.jsonl.gz" \
  --prior-q22 "$Q22/artifacts/manifests/frozen_q2_2_cohort.json" \
  --output "$ART/manifests/frozen_visual_grounding_cohort.json"

# Stop only the user-owned burner processes on GPUs 2,3,6,7 before these two commands.
CUDA_VISIBLE_DEVICES=2,3,6,7 "$VENV" -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.run_subspace_capture_ep4 \
  --manifest "$ART/manifests/frozen_visual_grounding_cohort.json" \
  --output "$ART/raw_runs/subspace_ep4" --control "$ART/runtime/subspace_control.json"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.build_structured_probe_plan \
  --manifest "$ART/manifests/frozen_visual_grounding_cohort.json" --capture-run "$ART/raw_runs/subspace_ep4" \
  --signals "$ART/visual_grounding_signals.jsonl.gz" --plan "$ART/manifests/frozen_structured_probe_plan.json"

CUDA_VISIBLE_DEVICES=2,3,6,7 "$VENV" -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.run_structured_probe_ep4 \
  --plan "$ART/manifests/frozen_structured_probe_plan.json" \
  --output "$ART/raw_runs/structured_ep4" --control "$ART/runtime/structured_control.json"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.analyze_visual_grounding \
  --manifest "$ART/manifests/frozen_visual_grounding_cohort.json" --signals "$ART/visual_grounding_signals.jsonl.gz" \
  --plan "$ART/manifests/frozen_structured_probe_plan.json" --probe-run "$ART/raw_runs/structured_ep4" \
  --labels "$ART/structured_damage_labels.jsonl.gz" --gate "$ART/phase1_gate.json" --verdict "$ART/final_verdict.json"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.analyze_distributional_diagnostics \
  --labels "$ART/structured_damage_labels.jsonl.gz" --output "$ART/exploratory_distributional_diagnostic.json"

python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.render_reports \
  --baseline "$ART/baseline_audit.json" --manifest "$ART/manifests/frozen_visual_grounding_cohort.json" \
  --signals "$ART/visual_grounding_signals.jsonl.gz" --labels "$ART/structured_damage_labels.jsonl.gz" \
  --gate "$ART/phase1_gate.json" --verdict "$ART/final_verdict.json" \
  --distributional "$ART/exploratory_distributional_diagnostic.json" --reports "$ROOT/reports"
