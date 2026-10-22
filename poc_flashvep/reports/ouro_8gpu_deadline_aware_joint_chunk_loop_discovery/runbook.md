# Execution contract

The authoritative specification is `../ouro_8gpu_deadline_aware_joint_chunk_loop_poc_spec.md`.
The campaign uses eight independent single-GPU workers.
It does not implement TP, PP, or shared-instance DP serving.
The native-vLLM Ouro Loop PP program remains paused.

All tokens execute Static R4 with the original BF16 weights.
G changes a legal prefill yield boundary, not model depth.
Existing reports and raw results remain immutable.
The old SLO campaign used four workers, not eight.

## Evidence classes

- `LIVE_MEASURED`: actual natural generation or component execution.
- `DIAGNOSTIC`: instrumented or controlled numerical execution.
- `OFFLINE_COUNTERFACTUAL`: legal-state action analysis, not online speedup.
- `NOT_RUN`: no execution evidence.

## Current state

Phase 0 started after the complete specification review.
No adaptive policy result exists yet.
Each phase needs a separate gate decision.
The campaign must preserve all failed attempts.
