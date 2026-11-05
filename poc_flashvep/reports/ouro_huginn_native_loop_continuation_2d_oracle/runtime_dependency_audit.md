# Runtime and dependency audit

## Native execution boundary

The external runtime remains unchanged.
The candidate uses process-local hooks over the qualified numerical path.
Native Atomic Prefill uses one outer submission for a prompt region.
That submission can contain several recurrent graph replays.
It is not one indivisible R4 or R32 CUDA graph.

The native non-wavefront progress helper supports consecutive recurrent depths.
The Wavefront callback requires one recurrent depth per step.
The CPU replay reproduces that callback rejection for both models.
The candidate does not remove this guard.
The P worker uses the native non-wavefront callback.
The actual D worker does not always follow the requested non-wavefront setting.
An inherited probe hook forces D Wavefront scheduling when `D_local_probe_output_pacing=True`.
Every campaign template retained that flag, even without an active local probe.
The recorded native contracts show P Wavefront disabled and D Wavefront enabled.
Ouro's asynchronous path advances the tested grouped progress.
Huginn's synchronous path rejects more than one remaining recurrent depth.
This blocks Huginn Token suffixes, Loop28, and the deeper 2D action.
The one-depth Huginn Loop31 and 2D frontier remain qualified.

The closeout CPU regression invokes the actual inherited probe installer with fake stage objects.
The enabled flag reproduces the scheduler change.
The disabled flag preserves the requested native scheduler.
This test does not establish GPU correctness after disabling the flag.
No GPU experiment tested that configuration before the deadline.

A partial Prelude submission must preserve every prompt hidden row at its cut.
The candidate copies these rows only for an actual partial submission.
Huginn stores the recurrent state and fixed injection input in the two halves of each hidden row.
The candidate also prevents premature Huginn Coda-KV execution at an incomplete recurrent depth.
D executes that boundary work after the final recurrent pass.

## Retained failures and corrections

The first Ouro partial launch failed before model execution.
Two constructor wrappers captured the same reassigned Python variable.
The candidate now uses distinct constructor closure names.
A CPU regression checks those bindings.

The next diagnostic completed exact outputs but performed no partial migration.
The reused worker replaced the connector after the initial bootstrap hook.
The migration fingerprint gate rejected this run.
Exact output parity alone did not certify continuation.
The candidate now selects its connector at the actual PDEngine construction boundary.
A stable global factory supports ordinary Python pickling across stage owners.
The original failed configurations and logs remain intact.

An additional graph-shape hypothesis failed during stage construction.
It assumed that packed RecurrentGraphs exposed the bucket list of SteadyRecurrentGraphs.
The actual packed graph keys the exact query count and context geometry.
It already permits large, non-power-of-two Token prefixes.
The candidate removes the unsupported bucket-list change.
It retains the common native graph selector and cache-entry budget.
The discarded CPU-only bucket planner is not installed in GPU workers.
The failed startup runs remain failures. They are not performance measurements.

The closeout audit found another graph issue in successful Ouro 2D runs.
The four-entry cache can capture startup shapes before the primary single-request shapes.
Unseen shapes then use the eager fallback path.
The first medium start captured 4096, 3072, 8192, and 6144 rows on P.
The second and third starts did not capture the required 4096-row shape.
Their Service TTFT increased with the missing primary shape.
The serial primitive run missed both 4096 and 3072 and recorded no successful measured P graph calls.
Fallback totals include startup and must not become measured-only fallback counts.
Deterministic primary-shape warm-up is a future fix candidate, not a measured optimized result.

## Shared zero-offload and partial transport

The initial candidate retained an unoptimized snapshot descriptor builder for partial work.
The native baseline already used page/layer-packed descriptors.
The v5 candidate applies that generic packing to arbitrary valid depth-plane prefixes.
It uses the same implementation for Token, Loop, and 2D frontiers.
CPU tests compare every transferred source-to-destination byte mapping on small layouts.
They also test different Prelude, recurrent, and Coda prefix lengths.
GPU byte-identity and N32 qualification remain separate requirements.

The v5 zero-offload and partial paths share the registered continuation storage and native transport/ACK ownership.
A full-depth zero-offload request still uses the native full-prompt arithmetic submission.
It transfers only its valid final hidden row, not every registered hidden row.
The registered buffer can be larger than the payload.
Memory reports must distinguish those values.
The feature can select cuts per request before Prefill submission.
A mixed-cut batch may require an additional boundary for a full-depth request.
The zero-offload gate therefore evaluates the all-zero-offload case separately.
Mid-graph preemption and a new online cut policy are not implemented.

## One-handoff 2D dependency

The P partition is a downward-closed staircase.
P computes all tokens through depth c and prefix k through the remaining depths.
D computes the remaining suffix through the remaining depths.
A P-owned tile must not depend on a D-owned predecessor before the single handoff.
CPU tests reject partitions that violate either token causality or previous-depth dependence.
This structural proof does not establish GPU correctness or performance.

For the current implementation, a nonzero-c frontier transfers the full retained prompt hidden buffer.
Only its suffix is required for D continuation.
The redundant prefix rows are not silently removed from measured payload accounting.
Token-only Huginn migration also transfers the exact canonical suffix initialization state.
All valid per-depth KV prefixes remain part of the transfer.

Equal recurrent token rows imply equal recurrent KV byte counts in the dense fixed-depth layout.
They do not imply equal Attention work, boundary work, or GPU duration.
Huginn Coda-KV work can materially change the D cost.
Recurrent row fractions must not become measured FLOP or speedup fractions.

## Reference engine and prior work

The reference LoopServing engine uses a different device-resident controller and pooled KV ABI.
Its GPU-side allocation, release queues, row routing, and graph controllers are not drop-in vLLM-RLT components.
Its README and pooled-KV design also describe early-exit-specific state rules.
This campaign does not import those rules or its weights/runtime.
It reuses the current native graph, depth-indexed KV, NIXL, and ACK implementations instead.
The relevant design lesson is explicit state ownership and release after the final legal consumer.

FluidToken moves prompt-suffix work after a prefix-KV handoff.
The current Token connector is a FluidToken-style reproduction in vLLM-RLT.
It is not the official SGLang implementation or a complete reproduction of FluidPD's online policy.
The paper's pressure signals include both execution pressure and KV admission.
See [FluidPD, Section IV-B](https://arxiv.org/html/2610.06917v1).

An unrolled generic logical-layer split can reproduce the same recurrent boundary.
Both PD workers already retain all model weights.
No measured resident-weight saving follows from the Loop label.
Any 2D headroom result must remain separate from a claim of exclusive Loop-specific novelty.
