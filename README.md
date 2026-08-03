# MLLM-EP research snapshot

This repository is the first Git-managed snapshot of the local MLLM expert
parallelism research archive. It contains the source, measurements, and
decision record through **FlashVEP Phase 1b**. The current FlashVEP decision is
**NO-GO for Phase 2**; no FlashVEP Phase 2 implementation was started.

The archive had no `.git` metadata before this import, so this repository does
not claim to reconstruct earlier commit history. See
`docs/agent_handoff/import_provenance.md` for the import boundary and omitted
large artifacts.

## Suggested reading order

1. `docs/agent_handoff/phase1b_tp2dp2_discussion_20260803.md`
2. `poc_flashvep/reports/phase1b_tp2dp2_profile.md`
3. `poc_flashvep/reports/phase1b_expert_timing_audit.md`
4. `poc_flashvep/results/baseline/gate_phase1b_tp2dp2_vision896.json`
5. `poc_flashvep/results/phase1b_tp2dp2_vision896/analysis_final.json`
6. `docs/flashvep_poc_spec.md` and the earlier TP4 records under
   `docs/flashvep_poc/`
7. `poc_flashvep/flashvep/instrumentation_phase1b.py` and the Phase 1b runner
   and analyzer under `poc_flashvep/scripts/`

For an independent review, copy
`docs/prompt/general_agent_flashvep_phase1b_review_prompt_ko.txt` into the
general agent after giving it access to this repository.

## Result at a glance

- Configuration: BF16, TP2/DP2/EP4/PP1 on physical GPUs 4-7.
- Actual runtime path: DPEP all-gatherv dispatch, 32 local Triton experts per
  rank, DPEP reduce-scatterv combine, then TP sequence all-gather.
- Expert boundary: 0.448672 ms median in-path. The earlier 0.447 ms value is
  plausible, but represents only the fused local-expert GEMM boundary, not a
  complete expert-parallel MoE stage.
- Selected-layer medians: dispatch 1.663 ms, expert 0.449 ms, mandatory
  combine drain 0.773 ms, layer 2.613 ms.
- Optimistic five-layer oracle: 1.07465x median, 1.08558x p90, 1.09963x max.
- Gate: NO-GO because the optimistic bound is below 1.10x/1.15x and profiler
  uncertainty is not cleanly separated from the remaining headroom.

Model weights, local caches, credentials, external upstream source snapshots,
and large Nsight binary/SQLite traces are intentionally not committed.
Derived evidence and checksums for omitted traces are retained.

> Note: documents and scripts with older `phase2*` names under `docs/archive/`,
> `scripts/`, or `outputs/` predate this FlashVEP PoC decision. They are
> historical MLLM-EP work and do not mean FlashVEP Phase 2 was started.
