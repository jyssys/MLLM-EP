# DeepEP Saturation/Latency Forensics

`DEEPEP_SATURATION_LATENCY: NO-GO`

## Scope and controls

Exact Qwen3-VL route artifacts (24 real-image requests × 48 layers) were replayed with current linear placement `expert_id // 32`. No model, routing, placement, or dynamic communication code was changed. DeepEP dispatch and combine were measured on the four logical ranks mapped to physical GPUs 1,2,3,4 (`CUDA_VISIBLE_DEVICES=1,2,3,4`) with BF16 hidden size 2048 and EP4; expert GEMM was excluded. Random BF16 communication payloads preserve route/layout/shape while avoiding model execution. Each rank injected the same artifact token rows to satisfy the collective contract; this is a bounded route-layout replay, not a claim about live DP source-token partitioning.

Fixed selection policy is stored in `poc_flashvep/deepep_revalidation/results/deepep_saturation_latency_20260827_171331/selection_policy.json`, including token matching ≤5%, similar/different S/I tolerances, and quartile regime boundaries. 24 pair-effect rows were measured; no post-hoc threshold changes were made.

## Matched results

| Effect | n | median absolute total-latency change | positive fraction |
|---|---:|---:|---:|
| S (matched other metric) | 4 | 1.69% | 25.0% |
| I (matched other metric) | 4 | 2.79% | 75.0% |

### Dispatch/combine breakdown

The matched effect is computed as (high metric − low metric)/low metric using the max-rank CUDA-event time per iteration. Negative values mean the designated high-S or high-I layout was faster.

| Comparison | Dispatch median Δ | Combine median Δ | Total median Δ |
|---|---:|---:|---:|
| S high vs low (I matched) | 0.12% | -5.00% | -0.44% |
| I high vs low (S matched) | -0.80% | 2.53% | 1.15% |

High-S/high-I versus the other quartile regimes interaction (total max-rank time): -0.53%

All-rank collective timing uses the maximum rank event per iteration, followed by a 20-iteration median/p25/p75/p95 summary. The 4-rank replay completed with 20 measured iterations per selected case and no DeepEP runtime errors.

## Modality diagnostic

Vision/Text labels are source labels within the same real-image request (`image_token_id == 151655` versus non-vision tokens). They were not used as a latency predictor. `modality_regime.csv` reports source-token S/I/rank-CV distributions; the figure uses modality-specific token subsets rather than request-level labels.

## Artifacts

Result directory: `poc_flashvep/deepep_revalidation/results/deepep_saturation_latency_20260827_171331`

Figures: `plot1_saturation_imbalance_latency.png`, `plot2_matched_case_effects.png`, `plot3_modality_regime_distribution.png`.

## Interpretation

The fixed gate found no repeatable ≥5% matched effect (or the required matched regimes were unavailable). Token/assignment volume remains the more defensible explanation in this bounded measurement.

### Limitations

DeepEP collectives were replayed with exact route IDs/layouts and deterministic random BF16 payloads, not live Qwen3 hidden states. The collective call is synchronous (`async_finish=False`) and no expert GEMM is included. If a required matched regime had insufficient artifact rows, it is explicitly marked unavailable rather than synthesized.

