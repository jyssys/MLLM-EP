# Part 1 baseline handoff check

## Authoritative bundle

The Part 2 run is pinned to
`poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/frozen_baseline_bundle.json`
(schema `mllm_ep_frozen_bundle.v2`).  The bundle hashes every reused asset; no
Part 1 artifact is overwritten by this experiment.

## Gate result

| Contract | Evidence | Part 2 status |
|---|---|---|
| Model | revision `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, d=2048, m=768, E=128, k=8, 48 decoder layers, patch16/merge2 | PASS |
| Image/model plumbing | Full public MMStar and ChartQA generation completed in Part 1 | PASS |
| Public reproduction | MMStar reference 71.985%, physical EP4 72.220%; ChartQA reference 85.600%, physical EP4 85.400% | `PUBLIC_PROTOCOL_PARTIAL` |
| Reference vs EP quality | MMStar EP-reference +0.400 pp, 95% cluster CI [-0.600,+1.467]; ChartQA -0.127 pp, CI [-0.519,+0.282] | local paired NI gate PASS |
| Physical EP4 | TP2/DP2/EP4/PP1, 32 experts/rank, nonzero DeepEP sparse dispatch/compute/combine | PASS |
| Low-resource semantics | 1/2-GPU 192/192 complete; exact-token 90.104%, parsed-answer 96.875% | PASS with numeric drift |
| EP2/EP4 latency simulator | EP2 median/P90 APE 12.962/22.728%; EP4 14.510/41.084% | `SIM_UNVALIDATED` |
| EP8 | workload structural verification and local 16-expert kernel only | `UNVALIDATED_SENSITIVITY_ONLY` |

Part 2 is therefore authorized for **physical EP4 discovery**.  Simulator
predictions cannot decide small candidate differences and are not labeled as
measured speedups.  EP8 output remains extrapolative.  Because the public
wrapper is only partially reconstructable, Part 2 can make paired local
quality statements but cannot claim official-protocol preservation.

## No-op and new-path requirements

The custom mixed-modality adapter is opt-in.  `R0/K8` returns the original
router tensors without cloning or reordering.  Actual EP4 smoke evidence also
checks the K4 path: visual rows execute four valid experts and four `-1`
sentinels while text/decode rows retain k=8.  The sentinel rows are rejected by
the sparse dispatch path and are counted separately from physical work.

The full R0/K8 discovery cell is the no-op parity measurement used by this
run; results are not borrowed from a changed resolution or a truncated trace.

