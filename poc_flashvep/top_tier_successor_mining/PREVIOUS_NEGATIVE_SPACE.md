# Negative-space map and limits of reuse

These historical results select controls; they do not establish failure of SERE,
Libra or MoDES. Fresh baseline sanity and transfer evidence remain mandatory.

| Prior direction | Reusable lesson | What it does NOT establish here |
|---|---|---|
| Fixed-shape DeepEP tails / selective wait | Moving a wait can improve dispatch without request E2E. Direct join is required. | No upper bound on expert reduction or replicated execution. |
| Fanout-aware serving predictor | Natural fanout added little after expert/rank features in the tested trace. | Does not test Libra's next-gate predictor or its replication planner. |
| Partial-expert speculative overlap | Output-vector error and created overlap must both be measured. | Hidden cosine is not task accuracy; top-k skipping is not SERE substitution or calibrated MoDES. |
| Modality TP/EP crossover | Match token count; verify actual backend activation. | Does not establish modality-independent approximation tolerance. |
| Active-expert fragmentation | Kernel/packing/runtime first-use can dominate isolated cost. | Assignment reduction is not linear latency reduction; no global negative for SERE. |
| Batch/wave studies | Wave duration can change while individual request latency is unchanged. | Wave-only gains cannot support successor E2E claims. |
| SM configuration sweep | Best-per-regime gains were small for that SM knob. | Not an upper bound on all communication/execution contracts. |
| Prior raw surrogate samples | Useful for debugging tensor identity and storage. | Self-including means or in-sample surrogates cannot support generalization. |

## Historical files read

- `reports/speculative_modality_overlap_discovery.md`
- `reports/final_ep_tail_go_nogo.md`
- `autonomous_moe_ep_night_discovery/KNOWN_NEGATIVE_SPACE.md`
- `speculative_modality_overlap_discovery/surrogate_probe.py`

## Mandatory safeguards

1. Separate original-model/original-system reproduction from Qwen-VL transfer.
2. Separate missing code, incompatible APIs and measurement patches from method failure.
3. Use held-out real task labels and confidence intervals; KL remains a proxy.
4. A full request has many decode invocations: never divide one 48-layer pass by
   full request time and present that as an E2E upper bound.
5. No candidate ranking until all three milestone records are populated honestly.
