# Trace summary

## Workloads and clean baseline

Both workloads use 32 submitted requests, mini32, generation budget 32, block length 32, threshold 0.9 and deterministic decoding.

| dataset | restarts | clean latency (s) | median (s) | NFE | median tokens/s | bounded quality |
|---|---:|---|---:|---:|---:|---:|
| GSM8K | 3 | 6.075 / 6.715 / 5.794 | **6.075** | 66 | 438.0 | 5/32 |
| HumanEval | 3 | 7.343 / 7.721 / 7.211 | **7.343** | 86 | 450.6 | 6/32 |

All 32 answers are bit-for-bit identical across the three clean restarts for each dataset. These bounded gen32 scores are only a substrate anchor; no approximate method was run and no quality improvement is claimed.

## Trace coverage

- GSM8K: 65 refinement waves × 31 MoE layers.
- HumanEval: 85 refinement waves × 31 MoE layers.
- Total layer-wave observations: 4,650.
- Rank-layer observations: 18,600.
- Temporal hidden/output observations: layers 1, 16 and 31.
- The measured prompt-prefill forward is excluded from refinement analysis.

Shape and stage traces independently reproduced exactly the same sequence IDs, block starts and remaining-mask trajectory for both datasets. This prevents a shared runtime-state drift from being mistaken for a shape/timing relationship.

## Observer tax

The selected stage traces took 22.379 s (GSM8K) and 23.939 s (HumanEval), versus clean medians of 6.075 s and 7.343 s. Shape traces are heavier still because they copy complete top-k and router arrays. Therefore their wall times are never performance evidence. Only internal same-device CUDA event durations and route/shape values are used.

## Component attribution upper bounds

Summed critical-rank CUDA events, normalized only for scale against clean E2E:

| dataset | router | dispatch | expert | combine | shared | gather | all measured MoE stages |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 6.43% | 12.07% | 29.47% | 6.41% | 2.98% | 2.17% | 59.52% |
| HumanEval | 7.22% | 13.32% | 29.55% | 6.66% | 3.50% | 2.55% | 62.81% |

These numbers show that a sufficiently strong MoE phenomenon could matter end-to-end. They do not establish that the measured stages are independently removable.

Primary artifacts: `BASELINE_CLEAN.csv`, `QUALITY_REPRODUCIBILITY.csv`, `TRACE_IDENTITY_CHECKS.csv`, `LAYER_WAVE_METRICS.csv`, `RANK_LAYER_METRICS.csv`, and `COMPONENT_E2E_UPPER_BOUNDS.csv`.

