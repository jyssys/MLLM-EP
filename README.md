# MLLM-EP research snapshot

This repository is the first Git-managed snapshot of the local MLLM expert
parallelism research archive. It contains the source, measurements, and
decision record through the **FlashVEP Batch 16/32 Quick PoC**. The current
FlashVEP decision is **HOLD**; no FlashVEP Phase 2 implementation was started.

The archive had no `.git` metadata before this import, so this repository does
not claim to reconstruct earlier commit history. See
`docs/agent_handoff/import_provenance.md` for the import boundary and omitted
large artifacts.

## Suggested reading order

1. `poc_flashvep/reports/batch16_32_quick_poc.md`
2. `poc_flashvep/results/baseline/gate_batch16_32_quick_poc.json`
3. `docs/flashvep_batch16_32_quick_poc_spec.md`
4. `poc_flashvep/results/batch16_32_quick_poc_20260804_131743/`
5. `docs/agent_handoff/phase1b_tp2dp2_discussion_20260803.md`
6. `poc_flashvep/reports/phase1b_tp2dp2_profile.md` and
   `poc_flashvep/reports/phase1b_expert_timing_audit.md`
7. `docs/flashvep_poc_spec.md` and the earlier TP4 records under
   `docs/flashvep_poc/`
8. `poc_flashvep/flashvep/instrumentation_phase1b.py` and the Phase 1b runner
   and analyzer under `poc_flashvep/scripts/`

For an independent review, copy
`docs/prompt/debate_agent_flashvep_batch16_32_review_prompt_ko.txt` into the
debate/general agent after giving it access to this repository.

## Result at a glance

- Configuration: BF16, TP2/DP2/EP4/PP1 on physical GPUs 4-7. Batch 16 split
  8/8 and Batch 32 split 16/16 across DP0/DP1.
- Actual runtime path: DPEP all-gatherv dispatch, 32 local Triton experts per
  rank, DPEP reduce-scatterv combine, then TP sequence all-gather.
- Valid Batch 16 selected-layer medians: dispatch 3.494 ms, expert 1.995 ms,
  combine drain 2.791 ms, layer 9.617 ms, and expert fraction 21.55%.
- Batch 16 existing/extended oracle: 1.175x/1.355x with 13.56% profiling
  overhead. Batch 32 ran without OOM, but its 61.81% overhead invalidates its
  stage numbers for gating.
- Gate: HOLD because Batch 16 is in the specified 20-25% expert-fraction band
  with little gain-overhead margin, while Batch 32 requires a lower-overhead
  revalidation.

Model weights, local caches, credentials, external upstream source snapshots,
and large Nsight binary/SQLite traces are intentionally not committed.
Derived evidence and checksums for omitted traces are retained.

> Note: documents and scripts with older `phase2*` names under `docs/archive/`,
> `scripts/`, or `outputs/` predate this FlashVEP PoC decision. They are
> historical MLLM-EP work and do not mean FlashVEP Phase 2 was started.
