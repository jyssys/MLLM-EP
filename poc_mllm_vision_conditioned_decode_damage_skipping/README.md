# Vision-conditioned decode damage / skipping PoC

This directory is an append-only bounded discovery implementation of
`MLLM_Vision_Conditioned_Decode_Damage_Estimation_and_Adaptive_Skipping_PoC.md`.
It reuses the frozen mixed-domain cohort and the validated TRUE EP4 capture
path.  Stage C--E are deliberately gated by held-out functional damage
prediction; a failed gate produces explicit `NOT_RUN_BY_STAGE_B_GATE`
artifacts rather than a speculative runtime result.

The exact commands used in the final run are recorded in
`artifacts/run_manifest.json` and `reports/final_summary.md`.

## Reproduce the compact analysis

Run from `/home/esjung/MLLM-EP-long-reasoning-signal` with `PYTHONPATH=.`.
The full exact command ledger, including the two append-only TRUE EP4 capture
runs, is in `artifacts/run_manifest.json`. The compact post-processing sequence
is:

```bash
python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_functional_damage \
  --run poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/functional_damage_v1 \
  --run poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/runs/functional_damage_v2_remaining \
  --plan poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/manifests/frozen_functional_probe_plan_v2.json \
  --local-labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/local_path_damage_labels_v2.jsonl.gz \
  --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz \
  --summary poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_damage_summary.json
python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.analyze_predictors \
  --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz \
  --contract poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/research_contract.json \
  --artifact-dir poc_mllm_vision_conditioned_decode_damage_skipping/artifacts
python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.build_ep8_structural \
  --labels poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/functional_path_damage_labels.jsonl.gz \
  --predictions poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/predictor_outputs.jsonl.gz \
  --output poc_mllm_vision_conditioned_decode_damage_skipping/artifacts/ep8_structural_projection.json
python -m poc_mllm_vision_conditioned_decode_damage_skipping.scripts.render_reports \
  --root poc_mllm_vision_conditioned_decode_damage_skipping
```
