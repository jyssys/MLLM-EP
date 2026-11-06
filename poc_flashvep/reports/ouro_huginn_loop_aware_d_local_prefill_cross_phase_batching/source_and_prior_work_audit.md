# Source and prior-work audit

## Scope

This PoC tests new-request D-local prefill. It does not continue the previous
Token/Loop migration campaign. A local request has no P-owned intermediate KV.
Native PD requests retain the qualified NIXL, commit, ACK, and release protocol.
All paths retain BF16, original weights, R4/R32, and the numerical fixes.

## Reused RLT mechanisms

The private runtime is `/home/esjung/external/looped-pd-abc-rlt`.
Its exact version and initial diff are in `state/preservation.json` and
`state/private_runtime.patch`. New hooks do not edit that runtime.

`StageEngine._add_request` validates a local prompt, sampling parameters,
context capacity, and full-depth prompt frontiers. The public D ingress rejects
ordinary prompt admission. The new connector command therefore uses the existing
validated stage-local ingress. It does not forge an imported partial state.

`MixedPolicy` already combines `RECURRENT` and `PREFILLING_RECURRENT` requests.
`Scheduler.take_phases` already builds row-specific request, position, and depth
metadata. `ModelRuntime` and the attention backend already address each KV plane.
Cross-phase, cross-depth batching is thus a reused runtime capability.
This PoC does not claim invention of that primitive.

The generic separate-execution control uses the same staged state representation.
Ready decode stages have priority. A prefill operation can still be nonpreemptible.
A new decode can become ready while that operation runs. Measured ITL gates remain
necessary. Strict ready priority can also starve prefill. Completion latency and
actual local work must therefore be reported.

The previous migration harness has an explicit no-mixed-execution guard.
Do not remove that guard from old experiments. Standalone microbenchmarks use
the native supported Mixed engine. Generic D-local controls select separate
phases. Full online cross-depth integration remains conditional on Phase C.

## Reference LoopServe compatibility

`poc_flashvep/ref_loopserve` is a vLLM V1 elevator runtime, not this RLT ABI.
Its README and `ouro_runner/loop_planner.py` describe depth-mixed waves,
non-loop boundaries, device-side KV funding, and policy costs.
`scripts/scheduler_supply_timeline.py` separates CPU queue snapshots from GPU work.
These design ideas are relevant. Its vLLM block-pool, device controller, page/run
maps, and request lifecycle cannot be substituted for RLT cache allocations.

The reference README also records an abandoned UT0-only cache that changed
attention semantics. That path is not eligible here. Its early-exit policies and
quality claims are outside this fixed-depth PoC. The reference source is read-only.

## Prior-work boundary

| Source | Relevant mechanism | Boundary for this PoC |
|---|---|---|
| [Load-Aware Prefill Deflection](https://arxiv.org/abs/2607.02043) | Whole-request D placement. Chunk sizing uses decode load, context, and latency predictions. | Generic D-local deflection is prior art. The local implementation is not the official system. |
| [FluidPD](https://arxiv.org/abs/2610.06917) | PPI/DPI pressure signals. Prefix KV handoff and D suffix execution. Worker-role elasticity. | This PoC does not implement FluidRole or partial-prefix handoff. A whole D-local request differs from FluidToken. |
| [CDB](https://arxiv.org/abs/2608.09444) | Queues at loop boundaries. Tied-core rows at different depths can share a batch. | More depth-compatible ready rows are already a CDB mechanism. Candidate-count growth alone is not a new contribution. |
| [ExactLoop](https://arxiv.org/abs/2611.05150) | Exactness-graded kernel, graph, KV, and transfer optimizations | Fixed-batch r0 changes differ from batch-composition changes. No approximate prefill, noise replacement, or early exit is imported. |

The deflection paper uses workload-aware prediction and also discusses a simulator
oracle. Its reported benefits do not qualify this PoC's runtime or latency model.
FluidPD explicitly accounts for D execution and KV pressure. Its partial handoff
still adds decode-side work. CDB explicitly uses tied weights across depths.
ExactLoop distinguishes exact fixed-batch changes from changed batching and noise.
These distinctions limit any new loop-specific claim.

## Falsification control

At fixed dense input values, row count, weight pointer, precision, and graph,
the recurrent-depth label is not an input to `Linear`. The matched same/cross
dense control therefore replays the same graph. Timing variation between those
labels is measurement noise, not a new operation. Real full-core comparisons
must also include valid per-depth attention, injection, packing, and KV writes.

An improvement over separate execution can be ordinary batching. An improvement
from allowing more ready depths can reproduce CDB. Neither establishes an
incremental advantage beyond the declared same-depth/generic control by itself.
The online integration gate remains unchanged.

## Startup failure and correction

Early all-local correctness attempts also deflected native graph warm-up cohorts.
Those cohorts wait for connector-owned transfers. A local request has no such
transfer. The host snapshots showed a queued PREFILL request with readiness false,
while the GPU was idle. This was a warm-up ownership mismatch, not device saturation.

The tested selection keeps all native warm-up cohorts on Native PD.
It runs a separate local warm-up after the native barrier closes.
The existing readiness guards remain active. Failed namespaces and their logs remain.
The CPU placement regression tests cover this distinction.
