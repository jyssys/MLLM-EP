# Training-Free EP-Aware Unmasking — final decision

Decision: **NO-GO at the discovery/economic screen for the current
LLaDA2.0-Flash B32, threshold-0.9, true-EP4 operating point.** No RL,
counterfactual full-rollout policy, compaction runtime, or variable-k change
was implemented.

## What was measured

The stock dense TP4/routed EP4/DeepEP-normal bridge ran on physical GPUs
4–7. All 256 routed experts remain owned in contiguous 64-expert/rank shards;
source rows were partitioned and remote DeepEP dispatch/reverse combine were
observed. Clean bounded-eight baseline was GSM8K 7.462 s/106 NFE/5 of 8 and
HumanEval 5.793 s/55 NFE/2 of 8. The complete GSM8K confidence/route trace
captured 518 valid token decisions. The HumanEval trace captured 247 valid
decisions before an observer-heavy Inductor error; a subsequent clean
HumanEval run succeeded. GSM8K observer timing was +30.2% versus clean,
while final answer/length records were identical; trace timing is not a
headline speedup substrate.

Across these 765 valid decisions, only 10.72% had even one unselected
alternative within 0.02 confidence of the selection cutoff, and the median
alternative count was zero. Near-tie candidates did have different routes:
all observed selected/alternative expert sets differed, and sampled-layer
rank-vector L1 differences had medians 10/12 assignments on GSM8K/HumanEval.
Nevertheless, a one-swap frozen-next-baseline-route screen reduced next-step
max-rank assignments by just 0.244% mean/0% median on GSM8K and
0.191% mean/0% median on partial HumanEval at delta 0.02. At relaxed delta
0.1 the combined mean was only 1.34%, without any semantic-safety proof.

The physical constraint is stronger than the motivating per-token story:
when the same number of positions is finalized, a live-only compacted next
step has the same number of fresh rows and, with unchanged top-k=8, the same
total routed expert assignments. An expensive position can be removed only
by leaving a different position live. The opportunity is a rank/expert
**shape change**, not total MoE-work removal. The current vanilla runtime
does not even remove finalized rows. Previous best-static matched EP4 route
replay found collision-versus-complement median differences of −0.14% and
−0.23% with the production fused path; `torch._grouped_mm` obtained +1.63%
EP-stage but best-pair versus random only +0.29% EP-stage. This is external
calibration, not a current-policy E2E measurement.

## Evidence boundary

`ONE_STEP_ORACLES.csv` is a frozen-baseline-route sensitivity calculation.
It is **not** a true future-aware counterfactual oracle: the baseline's next
hidden/router state for a finalized position differs from the alternative
MASK state. No alternative full trajectory, exact-quality E2E oracle,
measured post-Epoch run, future-cost predictor, or live policy was performed.
Therefore the direct request-level headroom is **unmeasured**, not 0.23%.
The NO-GO is the spec-authorized early stop because the semantic-safe choices
are sparse and the available shape-only physical screen is tiny. It is not
a theorem about other block sizes, decoders, EP degrees, or models.

## Required questions

| # | Question | Answer and boundary |
| ---: | --- | --- |
| 1 | Near-tie frequency? | 0.001/0.01/0.02 windows give combined 0.65/4.44/10.72% steps with any alternative. |
| 2 | Alternatives at cutoff? | Median zero; ≥2 at 0.02 only about 2.3% of decisions, ≥4 never. |
| 3 | Different expert routes? | Yes; 100% of the 148 observed 0.02 selected/alternative pairs have different sampled-layer expert sets. |
| 4 | Different rank EP cost? | Yes as assignment signatures: median L1 10/12 and remote difference 2/3 across five layers. Latency difference not established. |
| 5 | Best actual-latency metric? | Not established; no held-out C1–C5 latency calibration for unmask alternatives. Prior matched GPU replay warns max-rank alone is insufficient. |
| 6 | Temporal position-rank stability? | Median adjacent rank-vector cosine 0.976 GSM8K, 0.973 partial HumanEval. |
| 7 | One-step future-aware oracle? | True causal oracle not measured. Frozen-baseline-route one-swap max-rank screen averages 0.244/0.191% at 0.02; median zero. |
| 8 | Full-rollout gain? | Not measured after early gate. One-step numbers were not summed. |
| 9 | Exact-quality request E2E oracle? | Not measured; cannot claim ≥5%, ≥8%, or zero. |
| 10 | Tiny-quality-loss Pareto? | Not measured. Delta 0.05/0.1 are confidence-window diagnostics, not quality-safe operating points. |
| 11 | NFE change? | Stock NFE is 106/55; no alternative NFE measured. |
| 12 | Fresh rows versus fewer steps? | Same transfer count yields identical next live-row and top-k assignment totals under idealized live-only compaction; no row-saving credit. |
| 13 | Vanilla realizability? | Vanilla recomputes full block rows, so no accepted-row payload removal; shape effects were not replayed for the policy. |
| 14 | Liveness-aware opportunity? | Post-Epoch direct E2E not measured; algebraically no total-row reduction for same transfer count, only route-shape opportunity. |
| 15 | Future cost from current route? | Coarse rank stability exists, but actual next-wave marginal latency prediction was not validated. |
| 16 | Simple policy recovers ≥50%? | No policy/oracle recovery measured; gate was not met. |
| 17 | RL needed? | No RL justified; none trained. |
| 18 | Different from learned unmask policy? | A training-free confidence-admissible physical-cost tie break is conceptually distinct from RL, but measured headroom is absent. |
| 19 | MoE+EP specificity? | Route/rank signatures differ, but additional EP latency value over confidence-only or shuffled cost was not established. |
| 20 | Paper-level 8–12% E2E? | No credible evidence; direct E2E oracle unmeasured and the precursor screens are weak. |

## Next decision

Do not implement the unmask selector, RL objective, or compaction runtime for
this hypothesis now. A genuinely new attempt would need a different decoder
operating point with much more quality-validated confidence slack *and*
matched-route GPU evidence that rank/expert geometry materially changes
post-compaction service time. Otherwise this becomes a generic unmask/NFE
policy or repeats the prior wave-composition NO-GO.

Reproduction: `scripts/run_stock_ep4_atlas.sh`,
`scripts/run_stock_ep4_clean.sh`, `scripts/analyze_atlas.py`, and
`scripts/aggregate_atlas.py`. Raw run roots are in `results/gsm8k8_atlas_r1`,
`results/humaneval8_atlas_r1`, `results/gsm8k8_clean_r1`, and
`results/humaneval8_clean_r1`. Two CPU-only source-row/owner-map tests pass.
