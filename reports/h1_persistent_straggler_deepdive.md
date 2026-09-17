# H1 Persistent EP Straggler Deep Dive

## Verdict

**HOLD.** The persistent-straggler signal is real and the divisible-work upper bound is meaningful: EP4 has 6.11% aggregate routed-MoE headroom and EP8 has 12.37%. Exact-semantics replica oracles recover only 3.46% in the strongest early-persistent EP4 configuration and 5.86% in EP8. More importantly, global-static replication is stronger than early block adaptation, so the incremental dLLM persistence value is not method-level.

## Evidence boundary and invocation semantics

- Trace: 128 GSM8K requests at threshold 0.95, requests 0–63 discovery and 64–127 held out; accuracy 118/128 = 92.19%.
- `277,134` means **19 routed-MoE layers × 14,586 refinement forwards**, not independent generations.
- Rank expert time is predicted by target-EP single-H100 BF16 grouped-MLP replays; the complete EP2 routed-MoE predictor previously validated at 3.87% median and 9.40% P90 APE.
- EP4 and EP8 are `SIMULATED-EP4/8-EP2-CALIBRATED`, not physical runs.
- Replica routing preserves the selected expert ID and weight semantics. The aggregate trace lacks source×expert identity, so post-replica A/U traffic is a measured-dedup-rate estimate. Both compute-only and full-stage estimates are reported.
- The per-expert excess attribution allocates the calibrated critical-rank excess by exact expert-row share. It is not a per-expert GPU timestamp.
- No new model rollout was run. No true-EP2 replica replay was run because primary EP4 replication stayed below the prescribed 5% stage gate and EP2's perfect routed-stage upper bound itself is only 2.21%.

## Time-based straggler severity

Every cell below is ordered `P50 / P75 / P90 / P95 / P99 / max`.

| Target | Time max/mean | Time max/second | CV | Synchronization wait fraction |
|---|---:|---:|---:|---:|
| EP2 | 1.023 / 1.041 / 1.059 / 1.071 / 1.094 / 1.123 | 1.048 / 1.085 / 1.126 / 1.152 / 1.209 / 1.281 | 0.023 / 0.041 / 0.059 / 0.071 / 0.094 / 0.123 | 2.27% / 3.92% / 5.61% / 6.59% / 8.63% / 10.95% |
| EP4 | 1.074 / 1.122 / 1.159 / 1.171 / 1.255 / 1.265 | 1.053 / 1.109 / 1.153 / 1.178 / 1.210 / 1.246 | 0.064 / 0.089 / 0.112 / 0.116 / 0.168 / 0.187 | 6.91% / 10.86% / 13.75% / 14.58% / 20.29% / 20.94% |
| EP8 | 1.185 / 1.265 / 1.353 / 1.402 / 1.460 / 1.504 | 1.060 / 1.128 / 1.233 / 1.259 / 1.349 / 1.382 | 0.119 / 0.158 / 0.183 / 0.193 / 0.203 / 0.217 | 15.63% / 20.97% / 26.07% / 28.68% / 31.50% / 33.52% |

| Target | Critical-rank invocation distribution |
|---|---|
| EP2 | r0 74,551 (52.8%) / r1 66,524 (47.2%) |
| EP4 | r0 39,091 (27.7%) / r1 47,000 (33.3%) / r2 25,961 (18.4%) / r3 29,023 (20.6%) |
| EP8 | r0 24,055 (17.1%) / r1 15,045 (10.7%) / r2 7,365 (5.2%) / r3 30,046 (21.3%) / r4 19,666 (13.9%) / r5 14,302 (10.1%) / r6 21,507 (15.2%) / r7 9,089 (6.4%) |

Assignment imbalance is much larger (EP4 P50/P95 max/mean 1.285/1.624; EP8 1.721/2.668), but the calibrated grouped kernel compresses assignment imbalance. The time-based result is the latency-relevant one: EP4 still loses 6.91% rank-time at P50 and EP8 15.63%.

## Critical-rank persistence

| Target | Adjacent layer-local | Layer-global baseline | Cross-block | Matched random | Adjacent − global | Layer run P50/P90/P99/max | Forward adjacent/global |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP4 | 99.32% | 84.16% | 95.14% | 84.80% | +15.16 pp | 14/24/31/32 | 97.82% / 94.14% |
| EP8 | 98.83% | 80.71% | 95.85% | 81.12% | +18.12 pp | 13/23/31/32 | 100.00% / 100.00% |

The global bias is explicitly retained. At forward aggregate, EP4's 97.82% adjacent match is only +3.68 pp above the 94.14% global-majority baseline; EP8 is 100% for both, so EP8 forward-level identity persistence has no incremental within-block signal. Layer-local persistence has more incremental information (+15.16 pp EP4, +18.12 pp EP8), but cross-block persistence is already 95–96%.

| Target | Layer adjacent cosine | Matched-random cosine | Layer adjacent Spearman | Forward adjacent cosine | Forward adjacent Spearman |
|---|---:|---:|---:|---:|---:|
| EP2 | 0.999991 | 0.999861 | 0.908700 | 1.000000 | 0.799192 |
| EP4 | 0.999984 | 0.999655 | 0.992822 | 0.999999 | 0.988856 |
| EP8 | 0.999959 | 0.999094 | 0.993496 | 0.999998 | 0.992392 |

Cosine is near one even for matched random controls because a large shared/global load component dominates vector magnitude. Critical-rank identity and Spearman rank ordering provide the more discriminative persistence evidence.

## Critical excess concentration

| Target | Top1 P50 | Top2 | Top4 | Top8 | Experts for 25/50/75/90% P50 | High-imbalance states where top4 ≥50% |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 14.60% | 24.14% | 35.61% | 51.21% | 3 / 8 / 18 / 30 | 53.21% |
| EP8 | 23.82% | 41.20% | 57.37% | 75.72% | 2 / 3 / 8 / 14 | 68.22% |

EP4 is moderately distributed: eight experts are needed for 50% of excess at P50. EP8 is more concentrated: top four explain 57.37% at P50, making small replica budgets structurally more plausible.

## Early hot-expert prediction versus global popularity

| Target | K | Recall | Precision | Early excess-mass recall | Global excess-mass recall | Early increment |
|---|---:|---:|---:|---:|---:|---:|
| EP4 | 1 | 97.68% | 97.68% | 16.55% | 14.22% | +2.33 pp |
| EP4 | 2 | 97.81% | 97.81% | 26.72% | 23.56% | +3.16 pp |
| EP4 | 4 | 97.27% | 97.27% | 38.26% | 33.22% | +5.04 pp |
| EP4 | 8 | 97.31% | 97.31% | 53.59% | 46.26% | +7.33 pp |
| EP8 | 1 | 97.52% | 97.52% | 24.26% | 21.56% | +2.70 pp |
| EP8 | 2 | 97.37% | 97.37% | 38.38% | 33.78% | +4.59 pp |
| EP8 | 4 | 96.83% | 96.83% | 53.81% | 46.63% | +7.18 pp |
| EP8 | 8 | 96.69% | 96.69% | 72.35% | 62.30% | +10.05 pp |

Early refinements add 2.3–7.3 pp EP4 and 2.7–10.0 pp EP8 excess-mass recall over global popularity. This is real dLLM-specific information, but it does not translate into a better stage oracle than pre-resident global replicas.

## Perfect-balance oracle

| Target | Expert reduction aggregate | Routed-stage reduction aggregate | Routed-stage P50/P95/P99 |
|---|---:|---:|---:|
| EP4 | 8.11% | **6.11%** | 5.22% / 11.17% / 15.62% |
| EP8 | 17.16% | **12.37%** | 11.17% / 21.24% / 23.38% |

This is a divisible-work upper bound, not an implementable placement policy. It establishes that the earlier ~0.3% one-owner placement result did not measure total straggler headroom.

## Exact-semantics replica oracles — EP4

| Policy | Replicas/layer | Expert reduction | Compute-only stage | Full stage estimate | Candidate max/mean P50 | Setup-amortizing blocks |
|---|---:|---:|---:|---:|---:|---:|
| global static | 1 | 2.95% | 2.23% | 2.82% | 1.039 | — |
| global static | 2 | 3.81% | 2.87% | 3.52% | 1.033 | — |
| global static | 4 | 4.18% | 3.15% | 3.79% | 1.031 | — |
| global static | 8 | 4.21% | 3.17% | 3.87% | 1.031 | — |
| full-future block | 1 | 3.22% | 2.43% | 3.00% | 1.039 | — |
| full-future block | 2 | 4.02% | 3.03% | 3.61% | 1.035 | — |
| full-future block | 4 | 4.18% | 3.15% | 3.74% | 1.034 | — |
| full-future block | 8 | 4.18% | 3.15% | 3.74% | 1.034 | — |
| full-future envelope | 1 | — | — | 3.03% | — | — |
| full-future envelope | 2 | — | — | 3.72% | — | — |
| full-future envelope | 4 | — | — | 3.95% | — | — |
| full-future envelope | 8 | — | — | 3.98% | — | — |
| early refinement 1 | 1 | 2.99% | 2.25% | 2.78% | 1.040 | 84.60% |
| early refinement 1 | 2 | 3.73% | 2.81% | 3.35% | 1.037 | 84.80% |
| early refinement 1 | 4 | 3.88% | 2.92% | 3.46% | 1.036 | 81.52% |
| early refinement 1 | 8 | 3.87% | 2.92% | 3.46% | 1.036 | 73.10% |
| early refinements 1–2 | 1 | 2.78% | 2.09% | 2.58% | 1.042 | 82.33% |
| early refinements 1–2 | 2 | 3.46% | 2.61% | 3.11% | 1.039 | 82.95% |
| early refinements 1–2 | 4 | 3.60% | 2.71% | 3.22% | 1.038 | 79.63% |
| early refinements 1–2 | 8 | 3.60% | 2.71% | 3.22% | 1.038 | 69.65% |

## Exact-semantics replica oracles — EP8

| Policy | Replicas/layer | Expert reduction | Compute-only stage | Full stage estimate | Candidate max/mean P50 | Setup-amortizing blocks |
|---|---:|---:|---:|---:|---:|---:|
| global static | 1 | 3.99% | 2.87% | 3.92% | 1.147 | — |
| global static | 2 | 6.47% | 4.66% | 5.92% | 1.118 | — |
| global static | 4 | 7.12% | 5.13% | 6.42% | 1.111 | — |
| global static | 8 | 7.11% | 5.13% | 6.47% | 1.109 | — |
| full-future block | 1 | 4.51% | 3.25% | 4.31% | 1.143 | — |
| full-future block | 2 | 6.73% | 4.85% | 6.01% | 1.117 | — |
| full-future block | 4 | 7.19% | 5.19% | 6.37% | 1.112 | — |
| full-future block | 8 | 7.19% | 5.19% | 6.38% | 1.112 | — |
| full-future envelope | 1 | — | — | 4.32% | — | — |
| full-future envelope | 2 | — | — | 6.19% | — | — |
| full-future envelope | 4 | — | — | 6.69% | — | — |
| full-future envelope | 8 | — | — | 6.73% | — | — |
| early refinement 1 | 1 | 4.17% | 3.01% | 4.00% | 1.146 | 94.05% |
| early refinement 1 | 2 | 6.21% | 4.48% | 5.55% | 1.120 | 94.87% |
| early refinement 1 | 4 | 6.62% | 4.77% | 5.86% | 1.117 | 94.46% |
| early refinement 1 | 8 | 6.60% | 4.76% | 5.85% | 1.116 | 89.73% |
| early refinements 1–2 | 1 | 3.88% | 2.80% | 3.72% | 1.148 | 91.89% |
| early refinements 1–2 | 2 | 5.78% | 4.17% | 5.17% | 1.123 | 93.35% |
| early refinements 1–2 | 4 | 6.17% | 4.45% | 5.47% | 1.120 | 92.52% |
| early refinements 1–2 | 8 | 6.16% | 4.44% | 5.46% | 1.119 | 88.15% |

The strongest global-static results are 3.87% EP4 and 6.47% EP8. The best restricted full-future envelopes reach 3.98% and 6.73%; even future knowledge does not expose a large hidden replica opportunity.

The complete per-policy P50/P75/P90/P95/P99/max distributions for max-rank time, second-max time, mean, max/mean, max/second, CV, wait fraction, and dispatch/expert/combine/stage component totals are retained in `reports/h1_replication_oracle_summary.json`.

## Replica cost

- Exact checkpoint shapes: gate `[512,2048]`, up `[512,2048]`, down `[2048,512]`, BF16.
- One expert instance: 6,291,456 bytes = 6.00 MiB.
- Eight replicas per each of 19 routed layers: 956,301,312 bytes = 0.891 GiB total pool, before allocator/alignment overhead.
- If replica destinations are balanced, that budget averages 228.0 MiB/rank at EP4 or 114.0 MiB/rank at EP8; the conservative one-rank concentration bound is the full 912.0 MiB.
- Early-1–2, budget 4 setup-charged break-even P50: 4.92 future refinements EP4 and 2.64 EP8.
- The setup charge is an evidence-derived sensitivity: endpoint payload time at measured 85.51 GB/s plus one measured EP2 dispatch-startup charge per affected layer. It is not a direct expert-weight-copy measurement.

## Decision

- H1A severity: **PASS**.
- H1B persistence: **PASS** for EP4/EP8.
- H1C concentration: **PASS**, especially EP8.
- H1D perfect-balance headroom: **PASS**.
- H1E practical EP4 replica headroom: **WEAK** (<5%).

The correct conclusion is **HOLD / STRUCTURAL_HEADROOM_ONLY**. There is genuine imbalance and a nontrivial upper bound, but small-budget exact replication captures less than 4% EP4 stage time, and global static popularity captures more than the dLLM-specific early-persistence policy. No production method is justified yet.
