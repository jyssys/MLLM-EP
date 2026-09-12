# Production bottleneck table

## Clean versus instrumented evidence

Clean wall time is the performance source of truth. CUDA-event traces are used
for component proportions and geometry only.

| Trace | clean reference | instrumented | observer tax | E2E eligible |
|---|---:|---:|---:|---|
| EP4 micro8 block-only | 6.904 s | 8.248 s | +19.46% | no |
| EP4 micro16 block-only | 8.040 s | 7.817 s | -2.77% state variance | no |
| EP4 micro8 detailed | 6.904 s | 29.569 s | +328.27% | no |
| EP4 micro16 detailed | 8.040 s | 21.171 s | +163.32% | no |

## Representative EP4 microbatch-8 breakdown

The observer-light block trace attributes 53.06% of sequential block CUDA
time to MLP and 27.36% to attention. Mapping the detailed inner-MoE trace onto
that MLP share gives the following **analytical shares**, not direct clean E2E
measurements:

| Component | inner-MoE share | mapped block share |
|---|---:|---:|
| Router | 10.13% | 5.38% |
| Dispatch | 26.56% | 14.09% |
| Routed expert | 34.13% | 18.11% |
| Combine | 9.80% | 5.20% |
| Shared expert | 8.29% | 4.40% |
| Exact row gather | 11.10% | 5.89% |

At nominal M=256, 42.24% of active experts have at most four rows and rank-load
CV is 0.332. At the larger/variable-M trace, tiny groups remain 45.84% and CV
0.331. Larger waves therefore do not eliminate fragmentation or imbalance.

## DeepEP communication surface

DeepEP dispatch+combine median is 0.404 ms at global M=4, 0.257 ms at M=8,
0.410 ms at M=32, 0.424 ms at M=128, 0.325 ms at M=256, and 0.276 ms at
M=512. The non-monotonic, narrow range demonstrates a strong fixed/startup
component. Removing payload rows while retaining an all-to-all call is not
equivalent to proportional communication latency removal.

## Dominant practical bottleneck

The largest measured inefficiency is wrong execution granularity: EP4
microbatch 4 versus 8 costs 47.36% BCT. Once corrected, remaining novel
oracles are small. The next largest structural gap is liveness compaction, but
that is already Epoch's central contribution.
