# FlashVEP Phase 1b agent handoff

Date: 2026-08-03 KST

Scope boundary: audit and profile through Phase 1b only; Phase 2 not started

## Question under discussion

Does a real four-GPU TP2/DP2/EP4 DPEP path expose enough local-expert work to
justify implementing FlashVEP tiling/overlap, and was the prior 0.447 ms expert
measurement scoped correctly?

## What was established

The earlier TP4/DP1 run used `MoEPrepareAndFinalizeNoDPEPModular`. It had no
DPEP dispatch collective and ended with TP all-reduce, so it could not answer
the principal FlashVEP question. Its 0.447 ms value remains a valid CUDA-event
span around `_fused_experts`, but it is a fused invocation covering all work
assigned to 32 local experts on one rank. It is neither a per-expert latency nor
the end-to-end MoE latency.

The TP2/DP2/EP4 candidate was accepted for Phase 1b only after all four ranks
reported `MoEPrepareAndFinalizeNaiveDPEPModular` with
`AgRsAll2AllManager(allgather_reducescatter)`. Nsight linked the instrumented
ranges to NCCL broadcast-ring dispatch, BF16 reduce-ring DPEP combine, and TP
all-gather kernels. Thus the run exercises real inter-rank DPEP communication,
although dispatch is all-gatherv rather than a literal token-routed all-to-all
API.

The run used one real 799-token request on DP rank 0. DP rank 1 joined through
vLLM's native idle `START_DP_WAVE` path. Per layer, 6,392 real assignments,
8 TP-padding assignments, and 16 idle-DP dummy assignments reconcile exactly
to the 6,416 assignments seen by the gathered expert kernel. All 20 measured
requests returned token 1986 (`This`).

## Central measurements

Across 20 iterations, layers 0/12/24/36/47, and four ranks, the final stream
contains exactly 4,800 stage records with no null/error duration.

| Quantity | Median |
|---|---:|
| decoder layer | 2.613 ms |
| attention | 0.824 ms |
| norm/router | 0.188 ms |
| DPEP dispatch | 1.663 ms |
| local fused expert, critical rank | 0.449 ms |
| DPEP combine | 0.483 ms |
| TP sequence all-gather | 0.280 ms |
| complete combine drain | 0.773 ms |
| full MoE | 2.385 ms |

The Phase 1b in-path expert median is 0.448672 ms, independently reproducing
the prior 0.447 ms scale. An isolated layer-24 replay was faster at
0.2473/0.2539/0.2604/0.2498 ms on ranks 0-3 because it reuses warm dispatched
inputs and omits normal scheduling and handoff. This supports physical
plausibility but does not redefine the in-path stage.

The timestamp oracle retains decoder prelude/postlude, a nonzero first-tile
producer estimate, handoff gaps, and the full DPEP-combine-through-TP-all-gather
drain. It optimistically divides producer work into 400 uniform one-token tiles
and overlaps all remaining producer time with the complete expert window. For
the five selected layers, current time is 14.3296 ms median and the optimistic
lower bound is 13.3218 ms, yielding 1.07465x median, 1.08558x p90, and 1.09963x
maximum speedup.

The adjacent uninstrumented/instrumented request medians were 2,413.84 ms and
2,627.68 ms, or 8.86% profiler overhead. An earlier baseline at 3,825.62 ms was
preserved but excluded because concurrent system load changed; its cause is not
proven.

## Current position from Codex

**NO-GO for FlashVEP Phase 2 on this measured stack.** Dispatch and the
mandatory combine drain dominate the 0.449 ms expert window. Even an optimistic
uniform-tile oracle stays below 1.10x, no selected-layer median reaches 1.15x,
and the remaining headroom is not clearly larger than request-level profiler
uncertainty. No tiling, replay scheduler, custom kernel, or live overlap code
was implemented.

The recommended next task is not Phase 2. It is a standalone DPEP
collective/backend study aimed at reducing the 1.663 ms dispatch and 0.773 ms
combine drain before reconsidering tiling.

## Points for independent challenge

1. Is the class/runtime/Nsight evidence sufficient to call this a real DPEP
   dispatch/combine path despite its all-gatherv implementation?
2. Does the one-real-request plus idle-DP-wave setup change the workload enough
   to weaken the Phase 1b gate?
3. Is `producer_span / 400` a defensible deliberately optimistic first-tile
   bound, and does the oracle retain every non-overlappable gap exactly once?
4. Does the adjacent 8.86% overhead pair support stage-order conclusions while
   remaining too uncertain for a small achieved-speedup claim?
5. Does the untuned Triton baseline make the decision HOLD rather than NO-GO,
   even though faster expert execution would shrink the hideable window?
6. Is a standalone collective/backend study the single highest-information
   next experiment?

## Canonical evidence

- `poc_flashvep/reports/phase1b_tp2dp2_profile.md`
- `poc_flashvep/reports/phase1b_expert_timing_audit.md`
- `poc_flashvep/results/baseline/gate_phase1b_tp2dp2_vision896.json`
- `poc_flashvep/results/phase1b_tp2dp2_vision896/analysis_final.json`
- `poc_flashvep/results/phase1b_tp2dp2_vision896/stage_events_v2.jsonl`
- `poc_flashvep/results/phase1b_tp2dp2_nsys_224/nvtx_kernels_nvtx_kern_sum.csv`
- `poc_flashvep/flashvep/instrumentation_phase1b.py`
- `poc_flashvep/scripts/phase1b_tp2dp2.py`
- `poc_flashvep/scripts/analyze_phase1b_tp2dp2.py`
