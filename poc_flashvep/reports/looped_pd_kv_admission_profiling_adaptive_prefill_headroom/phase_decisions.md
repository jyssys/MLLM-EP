# Phase decisions

## Phase-by-phase decisions

| Model | Phase | Decision | Evidence scope |
| --- | --- | --- | --- |
| ouro | 0: internal correctness | PASS | Recorded token/state gates and full-depth audit only. Original-HF fidelity remains unmeasured. |
| ouro | A: admission | CONDITIONAL | Supported controls reduce specific binding limits. Long-context KV and slot limits remain workload dependent. |
| ouro | B: profiling | PASS_SCOPED | 1 matched profiles meet the predeclared wall-perturbation gate. Other profiles remain diagnostic. |
| ouro | C: workload/QPS search | COMPLETED_TESTED_GRID | Primary and secondary length sweeps remain visible. The chosen finite point has three paired starts when recorded. |
| ouro | D: same-start target | NOT_CONFIRMED | P compute gate=False. D SLO gate=True. D credit gate=True. |
| ouro | E: actual D borrowing | NOT_RUN_PREREQUISITE | No complete same-start compute/SLO/credit prerequisite supports the real-work probe. No synthetic point substitutes for it. |
| ouro | F: artifacts | COMPLETE_AFTER_VERIFIED_PUSH | The final response supplies the remote SHA. The publication receipt records the independent verification. |
| huginn | 0: internal correctness | PASS | Recorded token/state gates and full-depth audit only. Original-HF fidelity remains unmeasured. |
| huginn | A: admission | CONDITIONAL | Supported controls reduce specific binding limits. Long-context KV and slot limits remain workload dependent. |
| huginn | B: profiling | PASS_SCOPED | 4 matched profiles meet the predeclared wall-perturbation gate. Other profiles remain diagnostic. |
| huginn | C: workload/QPS search | COMPLETED_TESTED_GRID | Primary and secondary length sweeps remain visible. The chosen finite point has three paired starts when recorded. |
| huginn | D: same-start target | NOT_CONFIRMED | P compute gate=False. D SLO gate=True. D credit gate=True. |
| huginn | E: actual D borrowing | NOT_RUN_PREREQUISITE | No complete same-start compute/SLO/credit prerequisite supports the real-work probe. No synthetic point substitutes for it. |
| huginn | F: artifacts | COMPLETE_AFTER_VERIFIED_PUSH | The final response supplies the remote SHA. The publication receipt records the independent verification. |

## Why the target remains unconfirmed

Ouro admission relief changes available reservations and transfer draining. It does not create additional HBM or prove GPU saturation.
Huginn lighter inputs reduce per-request KV capacity. They also reduce Prefill compute, so the trade-off remains visible.
Several starts have legal D memory and acceptable Decode tails. Their P device-activity or service/backlog gate still fails.
Other starts have stronger pressure but binding D admission, a failed Decode tail, or a materially perturbed profile.
These conditions cannot be combined across starts into a positive result.
The measured kernel gaps are not exclusive CPU delay. A profiler does not identify the first historical blocked operation.
The experiment therefore does not prove that recurrence itself prevents Prefill-heavy serving.
A full Dynamic A campaign remains unjustified on this evidence.

## Implemented changes and explicit non-changes

All new source resides in `poc_flashvep/looped_pd_headroom_recovery/`.
The implementation uses supported native memory/slot/wire-credit controls. Native release and ACK dependencies remain intact.
The harness adds exact descriptor preparation, durable asynchronous evidence, and real fixed-depth graph warm-up.
The native CUDA-only profiler and selected-stage launcher collect the device evidence.
These controls are generic runtime changes, not a new Looped-specific scheduling contribution.
The external qualified runtime and its working diff remain unchanged.
No compact Huginn KV allocator, delayed-D-reserve protocol, new Dynamic A policy, or actual D borrowing path is implemented.
No model weight, BF16 precision, fixed depth, or attention equation changes.
LoopServing remains a read-only reference. The applicability matrix explains the API and execution differences.
