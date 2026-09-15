# Per-expert and per-tile hybrid oracle

## O1: actually executed subgroup hybrid

The measured hybrid partitions owner-packed rows by M_e, executes two exact
subgroups, then restores row order. Both directions and thresholds
`1/2/4/8/16` were tested. The crossover-directed variant uses vLLM fused for
tiny experts and PyTorch grouped for larger experts.

It never beats the strongest whole kernel in any of 102 real case-restarts.
Even the best threshold has a median penalty of **40.41%** on dense routes and
**47.35%** after compaction. The tiny `M_e=1` advantage cannot amortize the
second launch, index gather and scatter. Measured additional saving is 0% in
early, middle and late phases.

## O2/O3: one-launch upper bounds

The ideal scheduler is not built by serially summing single-expert latency.
Instead, O2 reuses the prior p99-robust critical-rank study: for every real
layer-wave it replaces expert latency with a nearby assignment-count lower
envelope while preserving assignments and useful FLOPs. This removes *all*
positive shape deviation—including noise—and is therefore a generous
one-launch persistent-scheduler ceiling.

| task | measured O1 | ideal dense O2 | credible dense O3 | post-compaction O2 | post-compaction O3 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 0% | **3.311%** | 1.655% | **2.119%** | 1.060% |
| HumanEval | 0% | **2.852%** | 1.426% | **1.134%** | 0.567% |

All values are additional request E2E over the strongest whole-kernel envelope.
O3 assumes a still-optimistic 50% capture of O2. Post-compaction numbers are
future-known live-row sensitivity, not measured Epoch. Even O2 fails the 5%
minimum gate, so CUDA implementation is forbidden by the working contract.

Figure: [14_per_expert_hybrid_oracle.png](../figures/14_per_expert_hybrid_oracle.png).

