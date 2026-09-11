# Semantic-first EP-critical-path top-k: final recovery checks

**Recovery decision: HOLD for one real EP8 validation; not GO.**

The 30% operating point has no observed benchmark-accuracy loss on the saved
32-request full-model cohort, and an exact offline assignment oracle grows
sharply with EP degree.  This invalidates an unconditional EP-scale NO-GO.
However, the EP8 result is a zero-overhead, router-risk-constrained **load
oracle**, not measured EP8 latency, TTFT, or quality for the oracle allocation.
It therefore justifies at most a bounded real EP8 validation—not a runtime or
paper claim.

No new GPU run was needed.  The quality rows and DeepEP replays are prior live
measurements; all EP scaling and layer selection results below are CPU-only
offline analyses.

## Evidence boundary

| Result | Evidence type | Permitted interpretation |
|---|---|---|
| 32-request greedy outputs | Measured full-model Qwen3-VL outputs | Bounded benchmark quality |
| EP4 replay, 30 reps/policy | Measured DeepEP operator latency | MoE-stage reduction and Amdahl projection |
| EP2/4/8/16 remap | Offline exact assignment/load oracle | Critical-rank work upper bound only |
| Historical EP8 route control | Real prior EP8 routes + offline allocation | Route-geometry sensitivity, not EP8 latency |
| Middle-layer subset | Measured three-layer replay + extrapolation | Layer-selection oracle, not integrated TTFT |

## 1. 30% benchmark-quality recheck

The earlier `31/32` was short-greedy **token-sequence identity**, not task
accuracy.  Re-scoring all 32 saved outputs with official-style ChartQA relaxed
numeric accuracy and the project's GQA answer-list exact metric gives:

| Policy | GQA | ChartQA | Aggregate | Accuracy |
|---|---:|---:|---:|---:|
| Stock | 11/16 | 14/16 | 25/32 | 78.125% |
| 30% semantic + EP refine | 11/16 | 15/16 | 26/32 | 81.250% |

Paired delta is +3.125 pp with a request bootstrap 95% interval of
`[0.000, 9.375]` pp.  There are zero stock-correct → policy-wrong flips and one
stock-wrong → policy-correct flip.  Consequently, the 30% point is
**benchmark-quality safe on this bounded cohort**; the sample is too small to
claim a quality improvement or population-level non-inferiority.

The only non-token-exact request is `chartqa_heldout_727`, asking for the
difference between the highest and second-highest points.  Gold is `2.87`;
stock generates `1.87`, while the 30% policy generates `2.87`.  This is not a
semantically equivalent spelling change—the approximation happens to repair a
stock error—so it must not be counted as a quality failure.

## 2. Offline EP-scale load oracle

### Definition

For each real captured route, the semantic 30% spatial schedule is fixed first.
The refinement MILP then chooses each vision token's exact prefix K in 1–8
while:

1. keeping text at K=8;
2. preserving the semantic schedule's exact retained assignment count;
3. limiting aggregate removed router risk to at most 1.05x semantic-only; and
4. minimizing maximum virtual-rank assignment load.

Experts use the requested 128-expert linear mapping,
`rank = floor(expert_id / (128/EP))`. All 216 virtual cases satisfy exact
assignment equality; maximum
risk ratio is 1.049998.  The optimizer has zero modeled runtime overhead and
does not establish task quality for its newly selected K map.

### Degree-isolation result on 54 real Qwen route captures

| EP | Cases | Semantic max/mean | Refined max/mean | Extra max work removed | Extra vs semantic | Extra reduction vs stock |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 54 | 1.078 | 1.000 | 35.0 rows | 7.20% | +5.37 pp |
| 4 | 54 | 1.218 | 1.003 | 69.5 rows | 17.10% | +13.28 pp |
| 8 | 54 | 1.550 | 1.035 | 82.5 rows | **29.01%** | **+21.31 pp** |
| 16 | 54 | 1.796 | 1.137 | 57.5 rows | 36.24% | +26.92 pp |

Here “extra vs semantic” is `(semantic max - refined max) / semantic max`;
“extra reduction vs stock” uses stock max-rank load as denominator.  Both show
a monotonic scale trend, not a 3% residual.

### Real historical EP8 route sensitivity

The repository contains an earlier real Qwen3-VL `TP2/DP4/EP8` trace with 128
experts and linear placement.  It lacks the benchmark-validated spatial
schedule, so it is a separate sensitivity control: 36 deduplicated
request/layer routes use a router-risk semantic allocation at 30% and the same
exact refinement constraints.

- Semantic max/mean: 1.400 median.
- Refined max/mean: 1.015 median.
- Extra max work removed: 183 rows median.
- Extra vs semantic: **27.33%** median (p10/p90 19.84/32.92%).
- Extra reduction vs stock: **+19.94 pp** median.

Thus EP-aware opportunity grows strongly in two route sources.  It clears the
user's 5–8% offline gate by a wide margin.  This does **not** imply 20–29% EP8
latency or TTFT: tiny-expert GEMM efficiency, communication, padding, and the
policy implementation cost are not modeled.  The result says only that a
large assignment critical-path upper bound exists at higher EP degree.

## 3. Middle/high-benefit-layer-only refinement

The previously quoted layer-24 increment of about 2.59 TTFT points subtracts
two medians that came from different requests.  With request-aligned pairs,
the measured replay evidence is:

| Representative layer | Requests | EP-only projected TTFT increment | Dispatch delta | Expert delta | Combine delta | Whole-MoE delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | +0.526 pp | +0.006 ms | -0.050 ms | -0.176 ms | -0.036 ms |
| 24 | 3 | **+1.554 pp median** | -0.012 ms | -0.124 ms | -0.289 ms | -0.105 ms |
| 44 | 1 | +0.573 pp | +0.039 ms | -0.087 ms | -0.147 ms | -0.039 ms |

The middle layer wins because its EP refinement removes more expert and
especially combine time; dispatch does not consistently improve.  One cell
image has a +4.76 pp increment, which also explains why unpaired aggregation
made the middle effect look larger.

In a coarse three-stratum oracle, applying semantic compression everywhere but
refinement only to the middle third (about 16/48 layers) gives:

- all-layer projected semantic+refine headroom: 12.213%;
- middle-only projected headroom: 11.847%;
- retention of **total** projected headroom: 97.0%, because semantic
  compression supplies nearly all of it;
- retention of the **EP-specific incremental** headroom: only **58.57%**;
- refinement exposure: 33.3% of layers, an unmeasured quality-risk proxy.

The broader assignment oracle does not confirm a sharply localized middle
band.  Across captured layers 4/12/24/36/44/47, the best one of six retains
19.69% of EP load-oracle mass, the best two retain 37.50%, and five are needed
for 86.73%.  Therefore a middle-only policy preserves nearly all *combined*
benefit only because Stage 1 dominates; it does not preserve most EP-specific
opportunity robustly.  No middle-only full-model quality run was performed, so
the reduced layer count cannot be presented as measured quality safety.

## Final answers

1. **Is the 30% point benchmark-quality safe?** **YES on the measured 32
   requests:** 25/32 → 26/32, with no corrected request becoming wrong.  This
   is bounded evidence, not a broad non-inferiority claim.
2. **Does EP-aware opportunity grow with EP scale?** **YES in the offline load
   oracle:** +5.37/+13.28/+21.31/+26.92 max-rank-reduction points for
   EP2/4/8/16, corroborated by +19.94 pp on historical real EP8 routes.  No
   EP8 latency/TTFT claim is made.
3. **Does middle/high-benefit-only refinement retain the gain?** **Only
   partially.** A middle-third oracle retains 97.0% of combined Stage-1+2
   projection but just 58.6% of EP-specific increment; six-layer load evidence
   is broad rather than middle-localized.

## Recommendation

The requested hard EP8 NO-GO condition is not met.  The appropriate status is
**HOLD**, narrowly for a real EP8 validation of the exact/risk-constrained
allocation.  Do not implement a variable-K runtime or custom kernel yet.  A
single bounded next experiment should apply a precomputed EP8 schedule in
Qwen3-VL, re-check benchmark quality, and measure whether the +21.31 pp load
oracle maps to at least 5–8 pp latency/TTFT after padding and communication.
Failure there would close the direction; success would be needed before GO.

Artifacts: `poc_semantic_first_topk/results/recovery_checks_20260911/`.
