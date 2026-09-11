# Result manifest

## Reused inputs

- Real expert-branch capture:
  `/home/esjung/MLLM-EP-vision-hetero-topk/poc_vision_heterogeneous_topk/results/vision_capture_fresh_20260911_0050`
- Prior clean request bottleneck:
  `poc_vision_heterogeneous_topk/results/bottleneck_clean_20260911_0041`
- Prior processed bottleneck atlas:
  `poc_vision_heterogeneous_topk/results/bottleneck_analysis_20260911_0051`
- Benchmark request manifest:
  `/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/top_tier_successor_mining_20260907_135950/data/requests.jsonl`

## Fresh offline and quality results

- `results/capture_oracles_full_20260911/`: 54 real image/layer logical
  allocation cases for router and actual-contribution risks.
- `results/semantic_sensitivity_eager_balanced16_20260911/`: GQA K=1..7
  task-conditioned sensitivity.
- `results/semantic_sensitivity_chartqa8_20260911/`: ChartQA K=1..7
  task-conditioned sensitivity.
- `results/combined_calibration_gqa8_chartqa8_20260911/`: exact
  multiple-choice-DP request and calibration-aggregate schedules.
- `results/proxy_quality_heldout32_20260911/`: held-out 32-request 0–30%
  selector/trivial-policy screen.
- `results/proxy_quality_heldout32_highdrop_20260911/`: held-out 32-request
  40–50% and fixed-K4 ceiling stress.
- `results/proxy_quality_heldout32_eprefine_20260911/`: held-out quality of
  the same-budget EP-refined schedules.

## Fresh actual DeepEP timing

- `results/deepep_replay_global_*`: global semantic schedule, layer and image
  robustness.
- `results/deepep_replay_trivial_*`: randomized selector/fixed-K attack.
- `results/deepep_replay_exactbudget_*`: exact assignment-count matched
  M=8177 primary replay.

JSONL event rows are intentionally ignored by Git; each run has a tracked
summary, and the derived audit tables preserve policy medians and metadata.

## Derived evidence

- `results/final_analysis/quality_rows.csv`
- `results/final_analysis/quality_summary.csv`
- `results/final_analysis/replay_summary.csv`
- `results/final_analysis/replay_l24_aggregate.csv`
- `results/final_analysis/summary.json`
- `results/final_analysis/plots/01_*.png` through `14_*.png`

## Evidence boundaries

- Sensitivity and quality results execute all experts and cannot establish
  runtime speedup.
- DeepEP replay establishes operator work/latency, not integrated request E2E.
- Projected TTFT uses a prior measured clean request share and remains an
  optimistic Amdahl estimate.
