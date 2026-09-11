# Data-derived candidate oracle comparison

## Candidate ranking

| Rank | Data-derived hypothesis | Evidence | Removable physical work | Optimistic request E2E | Correctness risk | Complexity | Prior-art overlap | EP2 decision |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | Perfect fresh-lane router reuse/prediction | router is 16.71 ms/request | upper bound is the router only | **2.693%** | high: fresh hidden/router changes | predictor + fallback | Epoch explicitly recomputes fresh gate; REFLEX/DES consume router | **NO-GO** |
| 2 | Delta-only destination/layout refresh | fresh destination set same 98.56% | layout metadata only; payload remains fresh | **0.0249%** | low for metadata | persistent plan + validation | Epoch caches block structure already | **NO-GO** |
| 3 | Stable fresh token-expert branch reuse | branch identity persists 59.62%, but values change | 0% exact; 0.998% stock at <=5% change | **0% exact; 0.0233% approximate** | severe without rollout | branch cache/versioning | Epoch caches decoded branches; this is fresh-lane residual | **NO-GO** |
| 4 | Early-layer/refinement-state selective refresh | layer 0 has a soft-stability pocket | 19.06% of fresh assignments in captured early cells at <=5% local change | **0.0232%** | approximate; one-step only | layer policy + branch cache | adjacent to Epoch/REFLEX but not identical | **NO-GO** |

All four numbers are perfect-future or stronger-than-realistic bounds.  Costs
for cache lookup, branch selection, gather/scatter, plan versioning, and
fallback are zero.  Failing the 5% gate under these assumptions is decisive.

## Causal interpretation

The measured causal chain is:

```text
same/coarse EP destination survives
    -> but hidden and gate values remain iteration-clock state
    -> exact expert identity often changes
    -> even persistent expert branches change substantially
    -> fresh payload and expert computation remain necessary
    -> only tiny metadata work is safely reusable.
```

The failed assumption was that route identity persistence implied expert-output
persistence.  The controlled branch decomposition holds token, layer, expert,
and adjacent iteration identity fixed and directly evaluates `E_e(h_t)` versus
`E_e(h_{t+1})`; the median expert-output change is 28.0% (weighted contribution
change 29.1%).  This rejects the premise
at the actual expert output, not merely at router IDs.

## Why no candidate proceeds to implementation

- Candidate 1 fails even if the *entire* router is free.
- Candidate 2 saves a measured ~4 us layout kernel but cannot remove dispatch
  bytes, because the hidden states change.
- Candidate 3 has no exact-reuse rows and too little tolerant-reuse work.
- Candidate 4 concentrates candidate work in layer 0, but fixed DeepEP/GEMM
  startup dominates and the layer occupies only 1/16 of the request's MoE
  stack.

There is therefore no reason to build a production branch cache, custom delta
dispatcher, or layer-refresh scheduler in this sprint.

## EP2 to EP4 transfer risk

| Quantity | What EP2 establishes | EP4 requirement |
|---|---|---|
| Temporal route/branch structure | directly observed | likely qualitative, but repeat |
| Live-vs-physical logical gap | decoder property | likely qualitative, but repeat |
| Dispatch/combine savings | not established for candidate | remeasure fanout/payload/backend |
| Load-to-latency curve | EP2, 32 experts/rank | recalibrate at 16 experts/rank |
| Absolute E2E gain | EP2 offline bound only | direct EP4 request validation |
| Correctness | local one-step change only | rollout and benchmark validation |

Normally a positive mechanism signal would be capped at `HOLD-FOR-EP4`.
Here no residual candidate reaches even 5% on EP2, so an EP4 validation is not
recommended for this work-removal direction.

See `analysis/plots/ep2_ep4_transfer_risk.png`.
