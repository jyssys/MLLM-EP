# Three Modality/Phase Pivot PoCs

## Final status

**ALL THREE NO-GO.** Fresh 4×H100 measurements falsify the required headroom;
no optimization or production scheduler should be implemented.

## Executive comparison

| Rank | Direction | Measured headroom | Projected E2E gain | Modality-specific evidence | Implementation complexity | Prior-art risk | Decision |
|---:|---|---:|---:|---|---|---|---|
| 1 | Vision/Text Attention × EP full-unit co-scheduling | best median eta 0.047 (unstable); best median wall saving 1.74% | <1% under observed full-MoE compatibility | No: max vision–text eta delta 0.040 | High (cross-request/layer scheduler) | High | **NO-GO** |
| 2 | Vision/Text/Decode phase-specific EP policy | 1.624% MoE latency; 0.00036% throughput oracle | 0.715% TTFT | Weak optima: vision sms16, text sms20, decode sms4 | Medium | Very high | **NO-GO** |
| 3 | Modality-aware layer/phase scheduling | 0.176% contention-corrected modeled makespan | 0.126% aggregate TTFT | No matched-shape cost/eta separation | Very high | High/directly adjacent | **NO-GO** |

The ordering only indicates which negative is least distant from its gate; none
is recommended for implementation.

## What was actually measured

- Two matched 2,363-token Qwen3-VL requests (vision-heavy with 2,340 vision
  tokens, and text-only) on TP2/DP2/EP4 DeepEP HT.
- Full stock layer-24 language Attention against fixed real routed DeepEP
  Dispatch, Expert, Combine and whole-MoE units: 4 warmups + 12 randomized
  samples per pair.
- All 48 layers' Attention/Dispatch/Expert/Combine/MoE times: three
  instrumented requests, separated from three clean requests.
- DeepEP communication SM and whole-request aggregation grids: 3 warmups + 10
  samples per point for vision prefill, text prefill and decode.
- Exact full logits/greedy tokens and replay tensors were unchanged.

Raw rank rows are intentionally gitignored because observer-heavy duplicated
flush payloads are multi-megabyte. Rank-deduplicated aggregate CSVs and all
analysis logic are versioned.

## Key findings

1. Full-size language Attention and EP stages are not complementary on this
   H100 path. Median eta stays between -0.027 and 0.047; p10 intervals cross
   zero for every pair.
2. Content modality does not create a resource regime at matched shape. Across
   48 layers, text vs vision attention differs by 1.4%, while the largest
   vision/text median-eta separation is only 0.040.
3. A zero-contention layer/phase oracle misleadingly promises 29.0%. Replacing
   that assumption with measured pair cost collapses the oracle to 0.176%, or
   about 0.126% of clean aggregate TTFT. This is the most important causal
   negative.
4. Phase-specific DeepEP SM optima differ, but best-static regret is 1.624% of
   MoE single-unit latency and 0.715% projected prefill TTFT.
5. Whole-request aggregation has a large generic throughput benefit, but a
   trivial static “fill available requests” cap recovers it. It is not a
   modality-aware policy opportunity.

## Evidence boundaries

- The observed request values are clean TTFT baselines: text 129.014 ms and
  vision 264.979 ms. Scheduling and EP-policy E2E numbers are explicitly Amdahl
  projections, not observed implementations.
- The pairwise fixed EP stage uses a real route/input replay and exact model
  weights, but is a diagnostic replay after the stock request MoE completes.
- Instrumented TTFT is 17–19% slower, so it is never used as removable latency.
- Stream priority could not be faithfully controlled through the installed
  runtime. That unsupported sub-axis cannot rescue or invalidate the already
  decisive best-static headroom failures.

## Final recommendation

Do not pursue any of the three pivots as a paper core on the present
Qwen3-VL/TP2/DP2/EP4/DeepEP-HT stack. The common failed assumption is that
modality labels imply different GPU resource occupancy at identical language
model shapes. Here the execution geometry, not content modality, determines
the resource regime. Any future revisit should require a genuinely different
kernel/backend geometry first—not another scheduler layered over these same
units.
