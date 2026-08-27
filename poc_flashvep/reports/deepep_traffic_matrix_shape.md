# DeepEP Traffic-Matrix Shape Forensics

`DEEPEP_TRAFFIC_MATRIX_SHAPE: HOLD`

## Scope and invariant controls

This is a communication-only synthetic replay. Each of four EP sources injects exactly N=256 or N=1024 tokens, each token routes to exactly two destination ranks with four balanced expert IDs per destination (top-k=8). Hidden payloads are deterministic random BF16 with H=2048. No model, expert GEMM, routing policy, placement, or dynamic communication code was used.

Family A (`balanced_spread` vs `pair_concentrated`) was asserted before timing to have identical source-row sums, destination-column sums, total incidence/assignment volume, S=0.5, and I=1.0. Only the number of source→destination pairs used by tokens differs. Family B (`destination_hotspot`) keeps token volume and S but intentionally skews destination columns as a diagnostic. The canonical matrices and invariant checks are in `traffic_matrices.json` and `invariant_check.csv`.

All 24 rank-label permutations were measured for both token scales. A permutation maps canonical source and destination labels consistently, preserving Family-A invariants while testing rank-label/topology dependence. Four logical ranks were mapped to physical GPUs 1,2,3,4 via `CUDA_VISIBLE_DEVICES=1,2,3,4`.

## Timing

Each case used 10 warmups and 50 measured iterations with a barrier before every iteration. CUDA events report the max rank. `layout_ms` is `get_dispatch_layout` only; `dispatch_only_ms` excludes layout; `combine_ms` is separate; `full_path_ms` spans layout through combine. Expert computation is absent. Raw rank samples are in `raw_timing.csv`; per-case summaries are in `case_summary.csv`; permutation effects are in `permutation_results.csv`.

## Family A result

| N | metric | median signed concentrated−balanced | median absolute | direction consistency | status |
|---:|---|---:|---:|---:|---|
| 256 | full path | 0.91% | 2.01% | 70.8% | NO-GO |
| 256 | layout | -0.65% | 1.02% | 70.8% | diagnostic |
| 256 | dispatch | 6.02% | 6.11% | 91.7% | diagnostic |
| 256 | combine | -5.43% | 5.43% | 100.0% | diagnostic |
| 1024 | full path | 14.06% | 14.06% | 100.0% | GO |
| 1024 | layout | 0.18% | 0.36% | 62.5% | diagnostic |
| 1024 | dispatch | 17.76% | 17.76% | 100.0% | diagnostic |
| 1024 | combine | 17.81% | 17.81% | 100.0% | diagnostic |

## Family B hotspot diagnostic

| N | median hotspot−balanced full-path change | median absolute | positive fraction |
|---:|---:|---:|---:|
| 256 | 3.45% | 3.47% | 83.3% |
| 1024 | 18.81% | 18.81% | 100.0% |

## Interpretation

The primary gate requires a ≥10% Family-A latency shift at both token scales with the same direction in at least 75% of all 24 permutations per scale. A 5–10% shift is HOLD; below 5% or inconsistent direction is NO-GO. The measured result is reported without changing those thresholds.

Overall gate: **HOLD**. Family-B hotspot is diagnostic only and cannot replace Family-A evidence.

## Limitations

Synthetic routes isolate communication geometry and do not represent live Qwen3 hidden-state timing. Each source rank owns its own synthetic route rows, but no expert GEMM is executed. CUDA-event timing uses synchronous DeepEP calls (`async_finish=False`); layout calculation and communication are separated. No real-trace Stage C was run because it is conditional on the primary Family-A gate being at least HOLD.

Result directory: `poc_flashvep/deepep_revalidation/results/deepep_traffic_matrix_shape_20260827_174530`

Figures: `plot1_iso_volume_traffic_shape.png`, `plot2_matrix_feature_vs_comm_latency.png`.
