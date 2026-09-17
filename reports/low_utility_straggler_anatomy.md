# Low-utility straggler anatomy

The primary population is the exact current-block route slots in the existing threshold-0.95 GSM8K-128 aggregate-v2 trace. Critical rank and excess use calibrated time over the complete physical workload. Thus excess shares are conservative shares of full physical excess. `current_router_weights` are the exact `topk_weight` coefficients returned by the official gate and multiplied into expert outputs; per-token sums are 2.5, so all threshold analyses use division by that sum.

| target | definition | P(low|critical) | P(low|noncritical) | enrichment | difference (pp) | ATTRIBUTED_EXCESS/full excess |
|---|---|---:|---:|---:|---:|---:|
| EP4 | slot_8 | 12.0350% | 12.7080% | 0.9470x | -0.6731 | 0.2065% |
| EP4 | slots_7_8 | 24.1027% | 25.4014% | 0.9489x | -1.2988 | 0.4132% |
| EP4 | slots_5_8 | 48.5934% | 50.6293% | 0.9598x | -2.0359 | 0.8312% |
| EP4 | bottom_10_global | 10.0122% | 9.9984% | 1.0014x | +0.0137 | 0.1781% |
| EP4 | bottom_25_global | 24.5827% | 25.1908% | 0.9759x | -0.6082 | 0.4309% |
| EP4 | bottom_50_global | 48.3738% | 50.7358% | 0.9534x | -2.3620 | 0.8313% |
| EP4 | mass_le_0.01 | 0.0002% | 0.0002% | 1.0791x | +0.0000 | 0.0000% |
| EP4 | mass_le_0.02 | 0.0053% | 0.0046% | 1.1474x | +0.0007 | 0.0001% |
| EP4 | mass_le_0.05 | 1.1483% | 1.0668% | 1.0763x | +0.0814 | 0.0207% |
| EP4 | mass_le_0.1 | 31.7848% | 32.9146% | 0.9657x | -1.1298 | 0.5542% |
| EP8 | slot_8 | 12.0400% | 12.6085% | 0.9549x | -0.5685 | 0.1869% |
| EP8 | slots_7_8 | 23.9849% | 25.2395% | 0.9503x | -1.2545 | 0.3726% |
| EP8 | slots_5_8 | 48.0219% | 50.4667% | 0.9516x | -2.4448 | 0.7470% |
| EP8 | bottom_10_global | 10.0328% | 9.9955% | 1.0037x | +0.0373 | 0.1586% |
| EP8 | bottom_25_global | 24.8205% | 25.0459% | 0.9910x | -0.2254 | 0.3900% |
| EP8 | bottom_50_global | 48.4846% | 50.3646% | 0.9627x | -1.8800 | 0.7571% |
| EP8 | mass_le_0.01 | 0.0003% | 0.0002% | 1.8369x | +0.0001 | 0.0000% |
| EP8 | mass_le_0.02 | 0.0058% | 0.0046% | 1.2638x | +0.0012 | 0.0001% |
| EP8 | mass_le_0.05 | 1.1037% | 1.0892% | 1.0133x | +0.0145 | 0.0175% |
| EP8 | mass_le_0.1 | 32.2449% | 32.6410% | 0.9879x | -0.3961 | 0.5056% |

The comparison is invocation-matched by construction: critical and non-critical routes come from the same layer/refinement invocation. Request-cluster bootstrap intervals and global-hot/random-rank controls are in the JSON artifact. Millions of slots are not treated as independent samples.

## Supplemental all-physical-row audit

The existing threshold-.95 GSM8K-32 heavy trace was audited over 3,823,104 evenly sampled physical token rows from all 59,736 layer/refinement invocations. This includes prompt, prior blocks, and current block.

| target | definition | critical | non-critical | enrichment | difference | sampled ATTRIBUTED_EXCESS |
|---|---|---:|---:|---:|---:|---:|
| EP4 | bottom 25% | 34.89% | 35.92% | 0.971x | -1.03pp | 35.28% |
| EP4 | mass <=10% | 40.96% | 42.20% | 0.970x | -1.25pp | 41.31% |
| EP8 | bottom 25% | 34.11% | 36.00% | 0.947x | -1.89pp | 33.80% |
| EP8 | mass <=10% | 39.84% | 42.35% | 0.941x | -2.51pp | 39.41% |

This audit reverses no conclusion: low-mass work is common, but is depleted—not enriched—on the critical rank.

The sampled low-mass-first curve needs 7.3283% of total EP8 router mass to cover 50% of modeled `ATTRIBUTED_EXCESS`. This is an attribution curve, not proof that removing those routes realizes the full excess reduction.
