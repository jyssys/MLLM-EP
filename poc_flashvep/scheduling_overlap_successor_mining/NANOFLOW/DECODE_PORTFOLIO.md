# Expanded decode portfolio and first-use boundary

`decode_volume_portfolio_20260909_v1`: 36 native engines, 3,888 requests,
B4/C128, B64/C128, B256/C32; 16 output tokens; graph and manual FFN plan1/2/4.
Three randomized independent restart blocks, same input IDs and warmup.

Only **2/27 whole-cohort paired comparisons** are token-exact. No alternative
plan satisfies three exact pairs in every workload. The multi-workload finite
envelope is **INSUFFICIENT / null**, not zero. Near-tie numerical sensitivity was
separately studied with same-plan and independent-HF controls; this is not proof
that every differing free continuation is semantically wrong. No differing
pair is used to promote a speedup or a method failure.

The first decode includes native plan construction/capture. Its median fraction
of target E2E ranges 58.12–93.77% across conditions. Required work is also inside
that interval: it is not a removable-cost estimate. Both decomposed times and
setup-inclusive E2E remain in the raw tables, with identity conservation checked
as `E2E = TTFT + sum(ITL)`. Steady ITL is not relabelled request completion.

The final same-input 96-output-token control tests setup amortization without
subtracting any target-path work. It remains a fixed-cohort diagnostic, not a
full warmed/asynchronous continuous-batching port.

## Long-output control completed

`long_decode_amortization_20260909_v1`: 18 engines / 612 requests, B4/B64,
graph/plan1/plan2, 96 output tokens and three randomized restarts. Only **1/12**
whole-cohort comparisons is output-exact; the envelope remains null.
First-decode share falls to roughly 30–71%, as expected from amortization.
Median steady ITL (raw diagnostic, not quality-certified performance):

| Cohort | graph | plan1 | plan2 |
|---|---:|---:|---:|
| B4/C128 | 6.459 ms | 6.235 ms | 8.727 ms |
| B64/C128 | 10.541 ms | 10.438 ms | 14.997 ms |

There is no observed split-overlap gain worth promoting here, but the differing
continuations and manual-plan scope prevent declaring the searched baseline
disproven. No timing is subtracted to manufacture a steady request-E2E benefit.
