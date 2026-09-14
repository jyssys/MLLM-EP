# Refinement EP shape atlas

## What is measured and what is derived

The source is a measured true-EP4 LLaDA2.0-Flash trajectory. For each source
rank, the exact pre-forward live mask filters the captured top-k expert IDs.
The four filtered rank-local lists are then joined. Routing identities are
measured; the compacted worklist is a future-known Epoch/FreshLane-like
sensitivity and is **not** a measured Epoch implementation.

The atlas contains 4,650 layer-wave rows: GSM8K has 65 waves x 31 routed-MoE
layers (2,015 rows), and HumanEval has 85 x 31 (2,635 rows). Every row satisfies
`token_expert_pairs = fresh_M * 8`; active-expert counts and compact pair counts
match the source trace.

## Fresh-M distribution

| task | p10 | p25 | p50 | p75 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 4 | 30 | 164 | 406 | 483 | 620 | 1,024 |
| HumanEval | 5 | 30 | 102 | 247 | 440 | 536 | 802 |

| task | very-small <=8 | small 9--32 | medium-small 33--128 | medium 129--512 | large >512 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 15.38% | 10.77% | 16.92% | 49.23% | 7.69% |
| HumanEval | 11.76% | 15.29% | 27.06% | 38.82% | 7.06% |

This is a genuine continuum rather than two endpoint modes: 65.6% of GSM8K
and 65.9% of HumanEval layer-waves lie in medium-small or medium bins.

## Phase structure

| task/phase | median fresh M | active experts | p50 rows/expert | tiny <=4 fraction | mean fanout | rank-load CV |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K early | 925.5 | 173.0 | 14.25 | 0.270 | 3.065 | 0.340 |
| GSM8K middle | 419.0 | 147.0 | 9.0 | 0.352 | 3.079 | 0.362 |
| GSM8K late | 35.5 | 71.5 | 2.0 | 0.685 | 3.053 | 0.381 |
| HumanEval early | 802.0 | 166.0 | 13.0 | 0.293 | 3.081 | 0.365 |
| HumanEval middle | 172.0 | 122.0 | 5.0 | 0.469 | 3.070 | 0.385 |
| HumanEval late | 15.5 | 45.0 | 2.0 | 0.794 | 3.012 | 0.454 |

Fresh M and rows/expert collapse across refinement, while destination fanout
stays near three ranks. Therefore liveness compaction would indeed expose a
high-mass changing physical EP workload. This establishes the workload
characterization premise, not the new-kernel premise.

Figures: `01_fresh_m_over_refinement.png`, `02_fresh_m_histogram.png`,
`03_fresh_m_cdf.png`, and `17_shape_class_mass.png` under `figures/`.
