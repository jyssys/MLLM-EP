# FlashVEP Phase 1b: TP2/DP2/EP4 profile

Date: 2026-08-03 KST  
Decision: **NO-GO for FlashVEP Phase 2**

## Audit scope and artifact mapping

The requested prompt, `docs/flashvep_poc_spec.md`, prior TP4 reports, gate
JSON, status, instrumentation, runners, and route/timing helpers were read in
full before editing. All requested paths existed under the current archive
root `/home/esjung/MLLM-EP`; no path remapping was needed. This directory has
no Git metadata, so a commit/worktree audit was unavailable.

The prior TP4/DP1 result is valid for its measured path but not for DPEP:
`MoEPrepareAndFinalizeNoDPEPModular` had no dispatch collective, executed 32
local experts per rank, then used TP all-reduce. Its `T_dispatch=0` was a
structural fact. The original files were not overwritten.

## Actual Phase 1b configuration

- Physical GPUs: 4, 5, 6, 7; logical global ranks 0, 1, 2, 3.
- BF16, TP=2, DP=2, EP=4, PP=1, sequence-parallel MoE enabled.
- TP groups: `[0,1]`, `[2,3]`; DP groups: `[0,2]`, `[1,3]`; EP group:
  `[0,1,2,3]`.
- Expert ownership: rank 0 = 0-31, rank 1 = 32-63, rank 2 = 64-95,
  rank 3 = 96-127.
- Exact Qwen3-VL-30B-A3B-Instruct snapshot:
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Fixed input: one gray 896x896 image and `Describe this image briefly.`,
  799 prompt tokens (784 visual, 15 text/special), `max_tokens=1`, prefix
  caching off, 5 warm-ups and 20 measured iterations.
- Every measured real request returned token 1986 (`This`). DP0 owns the one
  real request. DP1 participates through vLLM's native `START_DP_WAVE` idle
  path and does not return a second output.

The optional vLLM `RoutedExpertsCapturer` was disabled because it assumes an
unsharded 79/80-token dimension and fails on the TP2 40-token shard. Route
shapes and counts were instead captured directly at the real post-dispatch
expert input. FlashInfer startup autotuning was also disabled after its dummy
batch exercised the same capturer mismatch. Runtime `moe_backend=auto` then
selected `TritonExperts`; this is explicit and the log warns that no tuned
H100 E=32,N=768 config exists.

## Proven runtime path

All four ranks reported:

- prepare/finalize: `MoEPrepareAndFinalizeNaiveDPEPModular`;
- all-to-all manager: `AgRsAll2AllManager` with
  `allgather_reducescatter`;
- local kernel: `TritonExperts`, 32 physical experts per rank;
- sequence-parallel finalization: enabled.

For the 896 case, each layer used DP/TP chunks `[400,400,1,1]`:

1. `dispatch_all_gatherv` gathered hidden states, top-k weights, and top-k IDs
   from `[400,400,1,1]` to 802 rows on every EP rank.
2. Each rank executed only its 32 local experts against the gathered routes.
3. `combine_reduce_scatterv` reduced and scattered the 802-row expert output.
4. The Qwen sequence-parallel block performed TP all-gather and truncated the
   one TP-padding token. The late TP all-reduce helper was called but was a
   no-op in this sequence-parallel configuration.

This backend implements DPEP dispatch with all-gatherv, not a literal
token-routed all-to-all API. It nevertheless performs real inter-rank DPEP
dispatch and reduce-scatter combine; it is not the prior NoDPEP/TP-only path.

Nsight Systems 2024.6.2 mapped the selected NVTX ranges to these kernels:

| stage | observed GPU kernel |
|---|---|
| DPEP dispatch | `ncclDevKernel_Broadcast_RING_LL` |
| DPEP combine | `ncclDevKernel_Reduce_Sum_bf16_RING_LL` |
| TP sequence combine | `ncclDevKernel_AllGather_RING_LL` |

The trace and NVTX-to-kernel CSV are in
`poc_flashvep/results/phase1b_tp2dp2_nsys_224/`.

## Routing and local workload

The real request has exactly 6,392 route assignments per layer:
6,272 visual plus 120 text/special. TP padding contributes 8 more and the two
idle-DP dummy tokens contribute 16, so the physical gathered kernel sees
6,416 assignments. These categories reconcile exactly in every selected
layer.

| layer | rank-0 assignments / max batch | rank-1 | rank-2 | rank-3 |
|---:|---:|---:|---:|---:|
| 0 | 1,371 / 283 | 2,010 / 554 | 1,887 / 470 | 1,148 / 303 |
| 12 | 900 / 209 | 2,348 / 465 | 1,246 / 455 | 1,922 / 374 |
| 24 | 1,194 / 353 | 1,725 / 327 | 1,528 / 322 | 1,969 / 363 |
| 36 | 1,945 / 300 | 1,607 / 221 | 1,147 / 232 | 1,717 / 257 |
| 47 | 2,119 / 419 | 1,291 / 193 | 1,307 / 173 | 1,699 / 359 |

The layer-24 isolated fused-expert microbenchmark measured rank medians of
0.2473, 0.2539, 0.2604, and 0.2498 ms after 20 warm-ups over 100 iterations.
See `phase1b_expert_timing_audit.md` for the complete scope audit.

## Selected-layer stage breakdown

Coverage is complete: 20 iterations x 5 layers x 4 ranks x 12 stages = 4,800
records, with no null/error duration. Values below are medians of per-iteration
critical-rank quantities; they are not sums of independently aggregated stage
medians. `T_combine` is the timestamp span from DPEP combine start through TP
all-gather end and therefore includes the required combine drain.

| layer | T_layer | T_attention | T_norm/router | T_dispatch | T_expert_max | DPEP combine | TP AG | T_combine drain | full MoE | oracle | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.772 | 1.045 | 0.222 | 1.727 | 0.450 | 0.499 | 0.283 | 0.797 | 2.652 | 3.575 | 1.055x |
| 12 | 2.602 | 0.820 | 0.187 | 1.673 | 0.445 | 0.476 | 0.274 | 0.762 | 2.375 | 2.406 | 1.082x |
| 24 | 2.598 | 0.816 | 0.187 | 1.657 | 0.452 | 0.482 | 0.275 | 0.771 | 2.379 | 2.404 | 1.081x |
| 36 | 2.593 | 0.817 | 0.181 | 1.659 | 0.448 | 0.471 | 0.285 | 0.774 | 2.353 | 2.396 | 1.083x |
| 47 | 2.612 | 0.820 | 0.184 | 1.657 | 0.446 | 0.488 | 0.280 | 0.781 | 2.376 | 2.413 | 1.082x |

Across all 100 iteration/layer samples, medians were `T_layer=2.613 ms`,
`T_attention=0.824 ms`, `T_norm_router=0.188 ms`, `T_dispatch=1.663 ms`,
`T_expert_max=0.449 ms`, `T_combine=0.773 ms`, and
`T_full_moe=2.385 ms`. Median fractions were expert 16.95%, exposed
non-expert 83.05%, dispatch 63.86%, and combine drain 29.41%.

Critical layer ranks were predominantly rank 3 for layer 0 (16/20) and rank 1
for layers 12/24/36/47 (17/16/16/14 of 20). Critical expert rank was rank 0 in
15/16/16/16/15 iterations respectively; the remaining cases were rank 1.

## Timestamp-based optimistic oracle

The oracle is evaluated independently for the same iteration/rank/layer and
then takes the global critical rank. It retains the measured decoder prelude,
a nonzero one-token first-tile producer estimate, first-tile handoff, the
expert-to-combine gap, the complete DPEP-combine-to-TP-all-gather drain, and
the decoder postlude. The first-tile producer work is the measured producer
span divided by 400 local TP tiles. Only the remaining producer span is
allowed to overlap the complete expert window.

- Existing dispatch/expert/combine overlap is effectively absent: median
  measured overlap is 0.000695 ms.
- Median first-tile fill retained by the oracle is 0.1014 ms.
- The producer span is much longer than the 0.449 ms expert window, so the
  optimistic model can hide nearly all expert work but cannot save more than
  that short window.
- The 0.773 ms combine drain is itself longer than the expert window and is
  never hidden.
- The monolithic current attention/dispatch interface provides no observed
  tile-ready timestamp. Dividing the producer span into 400 uniform tiles is
  deliberately optimistic, so the result is an upper bound on plausible
  speedup, not an implementation forecast.

For the sum of the five selected layers, current time is 14.3296 ms median and
the optimistic lower bound is 13.3218 ms. Oracle speedup is **1.07465x median**,
1.08558x p90, 1.07659x mean, 0.00760 stddev, and 1.09963x maximum. No measured
selected-layer request reaches 1.10x, much less the 1.15x gate.

## Comparison with prior TP4/DP1

The old pass covered all 48 layers and the new pass covers five selected
layers; backend, TP semantics, and combine boundaries differ. The table is a
diagnostic comparison, not an apples-to-apples speedup measurement.

| Metric | TP4/DP1 previous | TP2/DP2 Phase 1b |
|---|---:|---:|
| actual dispatch collective | no | yes, all-gatherv |
| combine type | TP all-reduce | DPEP reduce-scatter + TP all-gather |
| T_attention | 0.908 ms | 0.824 ms |
| T_norm_router | 0.212 ms | 0.188 ms |
| T_dispatch | 0.000 ms | 1.663 ms |
| T_expert_max | 0.447 ms | 0.449 ms |
| T_combine | 0.808 ms | 0.773 ms drain |
| T_layer | 1.998 ms | 2.613 ms |
| oracle speedup | 1.0349x | 1.07465x selected-layer sum |
| profiler overhead | detailed +59.66%, lean +25.42% | +8.86% |

## Profiling overhead and blockers

The same-period no-profiler check was 2,413.84 ms median, 2,545.24 ms p90,
2,434.73 ms mean, and 66.99 ms stddev. The final selected-layer profile was
2,627.68 ms median, 2,691.82 ms p90, 2,640.62 ms mean, and 38.23 ms stddev:
**+8.86% overhead**, below the 15% limit.

An earlier no-profiler run in the same result directory measured 3,825.62 ms
median during changing concurrent system load. It is preserved but excluded
from the overhead denominator; the later adjacent baseline/profile pair is
used. The cause of that external sensitivity is not proven.

Remaining blockers are substantive:

1. dispatch (1.663 ms) and mandatory combine drain (0.773 ms) dominate the
   much shorter expert window (0.449 ms);
2. `AgRsAll2AllManager` exposes monolithic collectives, not tile-ready DPEP
   progress;
3. auto selected untuned Triton rather than a demonstrated strong FlashInfer
   CUTLASS baseline;
4. the oracle's 7.46% median headroom is smaller than the measured 8.86%
   request-level profiling overhead, although the CUDA-event stage ordering is
   still usable for the conservative gate;
5. the one-real-request external-DP run necessarily adds 8 TP-padding and 16
   native idle-dummy assignments per layer. These are separately accounted and
   do not change the output token.

## Gate and decision

- Real dispatch/combine EP collectives: **PASS**.
- Oracle median >=1.15x: **FAIL** (1.07465x).
- Repeated representative layers >=1.15x: **FAIL** (1.055-1.083x medians).
- Headroom clearly above profiler uncertainty: **FAIL**.
- Demonstrated tile-streaming interface: **FAIL**; only a theoretical uniform
  tile lower bound was evaluated.
- Strong baseline already overlaps most work: **PASS as a condition**; current
  stages are serialized, but the hideable expert window is still too short.

Because the median oracle is below 1.10x and the combine tail is longer than
the expert window, the Phase 1b decision is **NO-GO**. Phase 2 was not started.

The single recommended next task is a standalone DPEP collective/backend
study aimed at reducing the 1.663 ms dispatch and 0.773 ms combine drain before
any FlashVEP tiling implementation is reconsidered.
