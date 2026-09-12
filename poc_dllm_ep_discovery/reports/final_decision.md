# Final decision

## `CHARACTERIZATION-SIGNAL`

**One-sentence reason:** refinement creates a reproducible expert-shape/cost mismatch, but after matched controls and p99 robustness the best novel perfect request-E2E oracle is only 2.85--3.31%, and the post-Epoch residual only 1.13--2.12%.

## Gate audit

| strong-signal gate | result | pass? |
|---|---|---|
| logical work >=50% down, target physical latency <=20% down | live -78--80%; MoE -21--24%, expert -23--43% | no |
| live down, active support up, tiny fraction up | aggregate support falls; fixed-M support rises modestly but tiny fraction falls | no |
| shape explains latency much better than count | robust held-out expert RMSE -45.2% | **yes** |
| post-compaction residual >=10% E2E | 1.13--2.12% | no |

| candidate gate | measured result | decision |
|---|---:|---|
| current perfect fragmentation removal | 2.85--3.31% | KILL `<5%` |
| post-Epoch perfect residual | 1.13--2.12% | KILL `<5%` |
| feasible 50%-capture tiny path | 0.57--1.06% | KILL `<5%` |
| route-plan reuse | exact routes/outputs volatile | KILL |
| confidence-aware scope | causal premise absent and prior-art crowded | KILL |

## Required questions

1. **Which logical metric changes most?** Decision-live ratio: -80.0% GSM8K and -78.1% HumanEval early-to-late.
2. **Which physical metric moves unexpectedly weakly?** Whole measured MoE falls only 21.2% and 24.4%; dispatch falls about 23--26%.
3. **Does support broaden as liveness falls?** Not in raw phase aggregates. At fixed M=1024 it broadens modestly (+14.5%/+7.0%), with only +6.2%/+1.7% expert latency and no whole-MoE increase.
4. **Do rows/expert collapse?** Yes naturally: p50 15.75→5 and 15→3; mainly due to ready-pool/physical-M shrinkage.
5. **Does expert latency plateau?** It falls much slower than liveness on GSM8K, but misses the strict <=20% gate and falls 42.5% on HumanEval.
6. **Best predictor after controlling count?** Active-expert and rows/expert shape features; p99-trim held-out RMSE 0.02929 ms versus 0.05347 ms for count+layer.
7. **Is late fragmentation real?** Yes descriptively and after hypothetical compaction, but the causal liveness-driven broadening hypothesis does not survive fixed-M control.
8. **Would compaction worsen fragmentation?** Yes: late compacted <=4-row fractions reach 68.5%/79.4%, but this is modeled sensitivity.
9. **Post-Epoch residual >=8--10%?** No, 2.12%/1.13% perfect.
10. **Strongest novel oracle?** Current perfect fragmentation removal, 3.31%/2.85%; still killed.
11. **Does any candidate materially reduce total latency?** No oracle reaches 5%, let alone the 8% implementation gate.
12. **EP-specific?** Not established. A TP4 control was correctly not triggered because EP4 headroom failed. Treat shape sensitivity as generic fused-MoE characterization unless future evidence shows EP amplification.

## What is worth retaining

- Cost models for large dLLM MoE should include active-support/row-shape features; pair count alone is inadequate.
- Epoch-like liveness compaction should benchmark tiny-shape efficiency, but on this exact 4×H100 mini32 substrate the incremental request-latency opportunity is too small for a paper core.
- Coarse rank geometry can be temporally stable even when exact expert computation is not reusable.

## Do not pursue from this branch

Cross-wave expert queues, a bespoke 1--4-row kernel, confidence-conditioned top-k, exact temporal expert-output reuse, or a TP4/full-online port. Their current oracle or causal premise is inadequate.

