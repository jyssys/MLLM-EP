# Residual work breakdown

## Capture coverage

Three real cache-off generation trajectories were captured on true EP2:

| Request | Denoising forwards | Physical positions/forward | Branch-output coverage |
|---|---:|---:|---|
| expository | 19 | 128 | all 16 MoE layers |
| math | 29 | 128 | layers 0, 7, 15 |
| systems | 6 | 128 | layers 0, 7, 15 |

Together these contain 54 forwards, 864 physical layer invocations, 884,736
executed routed assignments, 104,448 token-transition-layer observations, and
118,665 comparable persistent token-expert branches.  At every layer/forward we
captured router logits, ordered top-k IDs/weights, and destinations; selected
layers additionally captured hidden states, actual MoE output, and manual
per-expert branch outputs from the rank-local BF16 weights.

Manual branch reconstruction agrees with the runtime result at mean-token
cosine >=0.9999946.  The maximum whole-tensor relative-L2 over captured
layer/requests is 0.542%, attributable to BF16 fused-GEMM accumulation order.
Both ranks had identical hidden state and routing (`max_abs=0`, route mismatch
count 0).

## What persists from t to t+1?

| State | Ordered top-k same | Top-k set same | Destination set same | Branch identity persists | Median hidden rel-L2 | Median MoE-output rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| All physical positions | 13.42% | 34.76% | 99.06% | 75.49% | 0.193 | 0.312 |
| Generation positions | 4.11% | 17.12% | 98.66% | 61.50% | 0.410 | 0.613 |
| Live generation | 1.91% | 13.84% | 98.66% | 61.91% | 0.361 | 0.521 |
| Newly decoded | 0.03% | 0.83% | 97.02% | 27.92% | 1.053 | 1.082 |
| Epoch-equivalent fresh generation | 2.42% | 13.87% | 98.56% | 59.62% | 0.406 | 0.583 |
| Stable decoded generation | 8.51% | 25.36% | 98.94% | 66.36% | 0.408 | 0.683 |

The surprising result is a hierarchy of stability:

```text
destination rank set (98.6% in fresh lane)
    >> individual expert branch identity (59.6%)
    >> full top-k set (13.9%)
    >> ordered top-k route (2.4%).
```

EP2's coarse rank map makes destination metadata appear nearly immutable even
when actual experts and contributions change.  That is useful for a metadata
oracle, but not evidence that expert evaluation can be skipped.

## Output-level persistence rejects the naive reuse premise

Among branches whose expert identity persists, the expert-output change has
median relative-L2 0.280 (p10 0.0463, p90 0.941).  The corresponding weighted
contribution change is 0.291 median (p10 0.0468, p90 0.976).  Candidate reuse
keeps the current router weight, so the more optimistic unweighted
expert-output criterion is primary.  Restricting to Epoch-equivalent fresh
persistent branches:

- <=0.1% change: 0%;
- <=1% change: 0.73% of persistent branches;
- <=5% change: 9.95% of persistent branches;
- <=10% change: 19.45% of persistent branches.

After including fresh branches whose expert ID changed and therefore cannot be
reused, only 0.448% of fresh assignments meet the 1% criterion and 6.10% meet
the 5% criterion.  Bit-identical branch reuse is zero.

Thus `same expert at t and t+1` is not a sufficient cache-validity condition.
Hidden state and router weight changes turn the same expert into materially
different decision-clock work.

## Layer structure

Early layer 0 is the only substantial soft-stability pocket: 48.91% of its
persistent fresh branches change by <=5%.  The rate falls to 12.86% at layer 1,
4.11% at layer 2, <=1.77% from layer 3 onward, and essentially zero through
middle/late layers.  Across broad bands, <=5% rates among persistent fresh
branches are:

| Band | <=5% branch change | Median expert-output change |
|---|---:|---:|
| Early (0--4) | 25.15% | 0.105 |
| Middle (5--10) | 0.17% | 0.534 |
| Late (11--15) | 0.07% | 0.372 |

This heterogeneity produces a legitimate child hypothesis (early-layer-only
reuse), but the economic oracle below kills it.

There is no common late-iteration convergence law.  Spearman correlation
between iteration index and the <=5%-change reusable share is -0.62
(expository), -0.06 (math), and +0.67 (systems).  The trajectories disagree in
sign, so “later refinement needs fewer fresh branches” is not a robust causal
policy in these samples.

## Figures

- `analysis/plots/mask_live_vs_physical_m.png`
- `analysis/plots/routing_stability_heatmap.png`
- `analysis/plots/branch_persistence_heatmap.png`

Observer-heavy traces are used only for these structural ratios, never as
clean request timing.
