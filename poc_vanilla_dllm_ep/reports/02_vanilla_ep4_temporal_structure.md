# Vanilla EP4 temporal/refinement structure

## Source contract

The released SDAR generator refines one 32-position block for 32 steps. Each
step runs the complete 48-layer decoder with a bidirectional-within-block
attention mask, samples target logits, and commits one highest-confidence
masked position under the fixed-threshold control. A final forward stores the
completed block's KV state. Vanilla does not use TEAM's decoded-position hidden
cache or expert restriction.

This gives three important boundaries before interpreting measured temporal
similarity:

1. A previously accepted position can still affect every remaining masked
   position through within-block attention. Route persistence is therefore not
   proof that its expert output can be reused exactly.
2. A conventional autoregressive speculative-verification mask is invalid for
   a sequence of bidirectional refinement states. A multi-step oracle is not a
   live exact method unless it supplies a SimSD/trajectory-style verification
   contract.
3. Eliminating recomputation for dead/accepted rows is Epoch's primary space,
   not independent novelty for this PoC. The surviving question is whether
   required *global* EP refinements can be reduced in frequency or scope.

## Measurement status

The final values in this report are populated from the separate observer-heavy
EP4 trace. Clean request latency comes from Stage 0 and is never replaced with
the tracing wall time.


## EP4 observer-heavy result

| Lag | Ordered top-k equal | Top-k set equal | Stable branches | Rank-destination set equal | Top-8 hot-expert Jaccard |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.27% | 33.44% | 82.18% | 80.02% | 76.85% |
| 2 | 9.35% | 27.40% | 77.42% | 76.34% | 72.12% |
| 4 | 6.49% | 21.15% | 70.26% | 71.52% | 64.34% |
| 8 | 3.68% | 13.98% | 59.12% | 64.82% | 51.01% |

Temporal routing and aggregate rank load are therefore highly persistent, but
the actual numerical state is not reusable as if it were unchanged:

| Lag | Hidden cosine / rel-L2 | Stable-branch output cosine / rel-L2 | Combined MoE cosine / rel-L2 | Rank-load cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.933 / 0.276 | 0.924 / 0.324 | 0.829 / 0.491 | 0.9971 |
| 2 | 0.900 / 0.351 | 0.895 / 0.399 | 0.771 / 0.609 | 0.9956 |
| 4 | 0.844 / 0.467 | 0.851 / 0.512 | 0.682 / 0.797 | 0.9923 |
| 8 | 0.748 / 0.655 | 0.781 / 0.698 | 0.550 / 1.118 | 0.9850 |

This is the main causal distinction in the trace: *which* experts/ranks are
needed is predictable, while *what values* they must compute changes
materially. It supports measuring exact weight-copy amortization, but it kills
the inference “stable route implies safe expert-output reuse.” It also means a
dense exact hidden delta has no natural byte advantage without a lossy codec.

The temporal run was intentionally observer-heavy: copying route/output tensors
to the host increased matched request time by 140.0%. Its timing decomposition
is not used as clean stage evidence; a separate CUDA-event run without tensor
capture supplies the stage shares.

An exact layout-reuse control further narrows the finding. At lag 1, the full
per-layer rank-count vector was exactly equal in only 0.144% of cases; the
per-token ordered owner vector was equal for 16.12%, and its owner multiset for
43.73%. Only 58.18% of top-k slots retained the same owner. Thus aggregate rank
load is predictable, but the exact dispatch plan usually changes. A cheap load
predictor is plausible; wholesale exact metadata/layout reuse is not.
